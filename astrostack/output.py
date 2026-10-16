"""Writing the stacked result out."""

import datetime

import numpy as np

BLOCK = 2880          # FITS is written in blocks of this many bytes
RECORD = 80           # each header keyword occupies this many characters


def _header_record(keyword, value, comment=''):
    if value is None:
        record = f'{keyword:<8}'
    elif isinstance(value, bool):
        record = f'{keyword:<8}= {"T" if value else "F":>20}'
    elif isinstance(value, (int, np.integer)):
        record = f'{keyword:<8}= {int(value):>20d}'
    elif isinstance(value, (float, np.floating)):
        record = f'{keyword:<8}= {float(value):>20.10G}'
    else:
        text = str(value)[:68].replace("'", '')
        record = f"{keyword:<8}= '{text}'"
    if comment:
        record = f'{record} / {comment}'
    return f'{record:<{RECORD}}'[:RECORD]


def write_fits(path, image, header=None, overwrite=True):
    """Write a 2-D or 3-D array as a FITS primary HDU.

    Written by hand rather than via astropy so the package has no extra
    dependency: the format is 80-character text records padded to 2880-byte
    blocks, followed by big-endian data padded the same way.

    :param image:  (H, W) or (n_planes, H, W) array; stored as float32
    :param header: extra keywords as {name: value} or {name: (value, comment)}
    """
    data = np.asarray(image, dtype=np.float32)
    if data.ndim not in (2, 3):
        raise ValueError(f'expected 2 or 3 dimensions, got {data.shape}')

    records = [_header_record('SIMPLE', True, 'conforms to FITS standard'),
               _header_record('BITPIX', -32, 'IEEE single precision'),
               _header_record('NAXIS', data.ndim, '')]
    # FITS axis order is the reverse of NumPy's
    for axis, length in enumerate(reversed(data.shape), start=1):
        records.append(_header_record(f'NAXIS{axis}', int(length), ''))
    records.append(_header_record('BZERO', 0.0, 'data are already physical'))
    records.append(_header_record('BSCALE', 1.0, ''))
    records.append(_header_record(
        'DATE', datetime.datetime.now(datetime.timezone.utc)
        .strftime('%Y-%m-%dT%H:%M:%S'), 'file creation, UTC'))

    for keyword, entry in (header or {}).items():
        value, comment = entry if isinstance(entry, tuple) else (entry, '')
        records.append(_header_record(keyword.upper()[:8], value, comment))
    records.append(f'{"END":<{RECORD}}')

    text = ''.join(records)
    text += ' ' * ((-len(text)) % BLOCK)

    # NaN marks pixels no frame covered. FITS carries IEEE NaN natively, so
    # they survive; tools that cannot handle them will show them as blanks.
    payload = data.astype('>f4').tobytes()
    padding = b'\x00' * ((-len(payload)) % BLOCK)

    mode = 'wb' if overwrite else 'xb'
    with open(path, mode) as handle:
        handle.write(text.encode('ascii'))
        handle.write(payload)
        handle.write(padding)
    return path


def read_fits(path):
    """Minimal reader, enough to verify what write_fits produced."""
    with open(path, 'rb') as handle:
        header, text = {}, ''
        while 'END' not in [line[:3].strip() for line
                            in [text[i:i + RECORD]
                                for i in range(0, len(text), RECORD)]]:
            block = handle.read(BLOCK).decode('ascii')
            if not block:
                raise ValueError('no END record found')
            text += block
        for i in range(0, len(text), RECORD):
            record = text[i:i + RECORD]
            keyword = record[:8].strip()
            if keyword in ('', 'END'):
                continue
            value = record[9:].split('/')[0].strip().strip("'").strip()
            header[keyword] = value
        shape = tuple(int(header[f'NAXIS{a}'])
                      for a in range(int(header['NAXIS']), 0, -1))
        data = np.frombuffer(handle.read(int(np.prod(shape)) * 4),
                             dtype='>f4').reshape(shape).astype(np.float32)
    return data, header


def write_tiff(path, image, bit_depth=16, low_percentile=0.5,
               high_percentile=99.9):
    """16-bit or float32 TIFF, for editors that cannot read FITS.

    32-bit preserves linearity exactly. 16-bit rescales to the integer range
    and therefore discards it, so use it for viewing rather than measuring.
    """
    import tifffile

    data = np.asarray(image, dtype=np.float32)
    if bit_depth == 32:
        tifffile.imwrite(path, data)
        return path
    finite = data[np.isfinite(data)]
    low, high = np.percentile(finite, [low_percentile, high_percentile])
    scaled = np.clip((data - low) / max(high - low, 1e-9), 0, 1)
    tifffile.imwrite(path, (scaled * 65535).astype(np.uint16))
    return path


def write_preview(path, image, target_background=0.20, low_percentile=1.0):
    """Stretched PNG for looking at.

    Linear data is unviewable -- the sky sits a few ADU above zero while
    stars saturate the top end -- so this applies the same midtone transfer
    function the diagnostics use. Never measure anything from this file.
    """
    import matplotlib
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    from .plotting import stretch

    data = np.asarray(image, dtype=np.float32)
    filled = np.where(np.isfinite(data), data, np.nanmedian(data))
    display = stretch(filled, lo_pct=low_percentile,
                      target_bg=target_background)
    if display.ndim == 3:
        display = np.moveaxis(display, 0, -1)

    height, width = display.shape[:2]
    figure = Figure(figsize=(width / 200, height / 200), dpi=200)
    FigureCanvasAgg(figure)
    axes = figure.add_axes([0, 0, 1, 1])
    axes.imshow(display, cmap=None if display.ndim == 3 else 'gray',
                origin='upper', interpolation='nearest', vmin=0, vmax=1)
    axes.set_axis_off()
    figure.savefig(path, dpi=200)
    return path
