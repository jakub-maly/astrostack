"""Writing the stack as a linear DNG.

DNG is a TIFF container, so it can be written directly. The variant used
here is a *linear* DNG: PhotometricInterpretation 34892 (LinearRaw) tells
the reader the data is already demosaiced but still linear sensor data --
which is precisely what a superpixel stack is. Lightroom and Darktable then
apply their full raw pipeline to it, so white balance, exposure and
highlight recovery all behave as they would on a camera file.

The colour matrix and white balance are copied from the source raw, so the
result renders with the same colour response as the originals.
"""

import struct

import numpy as np

# TIFF/DNG tag numbers
NEW_SUBFILE_TYPE = 254
IMAGE_WIDTH, IMAGE_LENGTH, BITS_PER_SAMPLE = 256, 257, 258
COMPRESSION, PHOTOMETRIC = 259, 262
MAKE, MODEL, STRIP_OFFSETS = 271, 272, 273
SAMPLES_PER_PIXEL, ROWS_PER_STRIP, STRIP_BYTE_COUNTS = 277, 278, 279
PLANAR_CONFIG, SOFTWARE, DATETIME = 284, 305, 306
SAMPLE_FORMAT = 339
DNG_VERSION, DNG_BACKWARD = 50706, 50707
UNIQUE_CAMERA_MODEL = 50708
BLACK_LEVEL, WHITE_LEVEL = 50714, 50717
DEFAULT_SCALE, AS_SHOT_NEUTRAL = 50718, 50728
COLOR_MATRIX_1, CALIBRATION_ILLUMINANT_1 = 50721, 50778
CFA_REPEAT_DIM, CFA_PATTERN, CFA_LAYOUT = 33421, 33422, 50711
CFA_PLANE_COLOR = 50710
ORIGINAL_RAW_FILE_NAME = 50827

BYTE, ASCII, SHORT, LONG, RATIONAL, SRATIONAL = 1, 2, 3, 4, 5, 10
TYPE_SIZE = {BYTE: 1, ASCII: 1, SHORT: 2, LONG: 4, RATIONAL: 8, SRATIONAL: 8}


def _rational(value, denominator=10000):
    return int(round(value * denominator)), denominator


class _Tiff:
    """Builds a single-IFD little-endian TIFF."""

    def __init__(self):
        self.entries = []

    def add(self, tag, type_code, values):
        if isinstance(values, (str, bytes)):
            payload = (values.encode('ascii') if isinstance(values, str)
                       else values) + b'\x00'
            self.entries.append((tag, ASCII, len(payload), payload))
            return
        values = list(values) if isinstance(values, (list, tuple)) else [values]
        if type_code in (RATIONAL, SRATIONAL):
            fmt = '<' + ('II' if type_code == RATIONAL else 'ii') * len(values)
            flat = [part for value in values for part in value]
            payload = struct.pack(fmt, *flat)
        else:
            fmt = {BYTE: 'B', SHORT: 'H', LONG: 'I'}[type_code]
            payload = struct.pack('<' + fmt * len(values), *values)
        self.entries.append((tag, type_code, len(values), payload))

    def build(self, image_bytes):
        self.entries.sort(key=lambda entry: entry[0])
        header = 8
        # entry count, the entries themselves, then the next-IFD pointer
        directory = 2 + 12 * len(self.entries) + 4
        overflow_start = header + directory

        # values longer than four bytes live after the directory
        overflow, offsets = b'', {}
        for tag, type_code, count, payload in self.entries:
            if len(payload) > 4:
                offsets[tag] = overflow_start + len(overflow)
                overflow += payload + (b'\x00' if len(payload) % 2 else b'')
        strip_offset = overflow_start + len(overflow)

        records = []
        for tag, type_code, count, payload in self.entries:
            if tag == STRIP_OFFSETS:
                payload = struct.pack('<I', strip_offset)
            field = (payload.ljust(4, b'\x00') if len(payload) <= 4
                     else struct.pack('<I', offsets[tag]))
            records.append(struct.pack('<HHI', tag, type_code, count) + field)

        return (struct.pack('<2sHI', b'II', 42, header)
                + struct.pack('<H', len(records))
                + b''.join(records)
                + struct.pack('<I', 0)
                + overflow
                + image_bytes)


def write_dng(path, image, colour_matrix=None, white_balance=None,
              camera_model='astrostack', software='astrostack',
              datetime_original=None, original_name=None,
              headroom=1.0):
    """Write a 3-plane linear RGB stack as a linear DNG.

    :param image:          (3, H, W) linear RGB, black level already removed
    :param colour_matrix:  3x3 XYZ(D65) -> camera matrix, i.e. rawpy's
                           `raw.rgb_xyz_matrix[:3, :3]`. Without it the
                           reader falls back to its own guess and the colour
                           will be wrong.
    :param white_balance:  camera multipliers, i.e. `raw.camera_whitebalance`
    :param headroom:       multiply the peak by this before scaling. 1.0
                           maps the brightest pixel to 65535; raise it to
                           leave room, lower it only if you accept clipping.
    :return:               (path, scale) where `scale` is the factor applied
                           to reach 16-bit; divide by it to recover ADU
    """
    data = np.asarray(image, dtype=np.float32)
    if data.ndim != 3 or data.shape[0] != 3:
        raise ValueError(f'expected (3, H, W) RGB, got {data.shape}')
    planes, height, width = data.shape

    # DNG carries unsigned integers, so the linear data is rescaled to fill
    # the 16-bit range. `scale` is returned and recorded so the mapping back
    # to ADU is not lost; negatives (real, from the sky pedestal) are clipped
    # because a raw format has no way to express them.
    # scale to the actual maximum, not a percentile: clipping the brightest
    # stars is exactly what a raw file must not do
    peak = float(np.nanmax(data)) * headroom
    scale = 65535.0 / max(peak, 1e-6)
    scaled = np.clip(np.nan_to_num(data, nan=0.0) * scale, 0, 65535)
    interleaved = np.moveaxis(scaled, 0, -1).astype('<u2')

    tiff = _Tiff()
    tiff.add(NEW_SUBFILE_TYPE, LONG, 0)
    tiff.add(IMAGE_WIDTH, LONG, width)
    tiff.add(IMAGE_LENGTH, LONG, height)
    tiff.add(BITS_PER_SAMPLE, SHORT, [16, 16, 16])
    tiff.add(COMPRESSION, SHORT, 1)                   # uncompressed
    tiff.add(PHOTOMETRIC, SHORT, 34892)               # LinearRaw
    tiff.add(SAMPLES_PER_PIXEL, SHORT, planes)
    tiff.add(ROWS_PER_STRIP, LONG, height)
    tiff.add(STRIP_BYTE_COUNTS, LONG, interleaved.nbytes)
    tiff.add(STRIP_OFFSETS, LONG, 0)                  # patched in build()
    tiff.add(PLANAR_CONFIG, SHORT, 1)                 # interleaved
    tiff.add(SAMPLE_FORMAT, SHORT, [1, 1, 1])         # unsigned integer
    tiff.add(SOFTWARE, ASCII, software)
    if datetime_original:
        tiff.add(DATETIME, ASCII, datetime_original)
    if original_name:
        tiff.add(ORIGINAL_RAW_FILE_NAME, ASCII, original_name)

    tiff.add(DNG_VERSION, BYTE, [1, 4, 0, 0])
    tiff.add(DNG_BACKWARD, BYTE, [1, 1, 0, 0])
    tiff.add(UNIQUE_CAMERA_MODEL, ASCII, camera_model)
    tiff.add(BLACK_LEVEL, SHORT, [0])                 # already subtracted
    tiff.add(WHITE_LEVEL, LONG, [65535])
    tiff.add(DEFAULT_SCALE, RATIONAL, [_rational(1.0), _rational(1.0)])

    if colour_matrix is not None:
        matrix = np.asarray(colour_matrix, dtype=float).reshape(3, 3)
        tiff.add(COLOR_MATRIX_1, SRATIONAL,
                 [_rational(v) for v in matrix.ravel()])
        tiff.add(CALIBRATION_ILLUMINANT_1, SHORT, 21)   # D65
    if white_balance is not None:
        multipliers = np.asarray(white_balance, dtype=float)[:3]
        multipliers = np.where(multipliers > 0, multipliers, 1.0)
        # AsShotNeutral is the camera-space colour of a neutral subject,
        # which is the reciprocal of the white-balance multipliers
        neutral = multipliers[1] / multipliers
        tiff.add(AS_SHOT_NEUTRAL, RATIONAL,
                 [_rational(v, 100000) for v in neutral])

    with open(path, 'wb') as handle:
        handle.write(tiff.build(interleaved.tobytes()))
    return path, scale


def colour_profile(raw_path):
    """Colour matrix and white balance from a source raw file.

    Copying these across is what makes the DNG render like the originals
    rather than with an arbitrary colour cast.
    """
    import rawpy

    with rawpy.imread(raw_path) as raw:
        matrix = np.asarray(raw.rgb_xyz_matrix, dtype=float)[:3, :3]
        balance = list(np.asarray(raw.camera_whitebalance, dtype=float))
    return matrix, balance


def interleave_bayer(planes, pattern, black_levels, white_level=None,
                     dtype=np.uint16):
    """Rebuild a full-resolution Bayer mosaic from four stacked sub-planes.

    Each plane must already be registered onto a common grid and stacked.
    Because the four sub-lattices sample the sky at fixed half-superpixel
    offsets, and every plane received the same transform, interleaving them
    reproduces the reference frame's own mosaic geometry.

    This keeps G1 and G2 *separate*, which is what makes the result honest:
    the two green sites hold independent measurements half a superpixel
    apart, exactly as a sensor produces, so a demosaicer can use their
    difference. Averaging them into one green and writing it to both sites
    would fabricate a correlation no sensor ever shows.

    The black level is added back so the data sits in native camera ADU: the
    pedestal exists precisely to keep read noise positive, and a stack's
    noise is smaller than a single frame's, so nothing clips.

    :param planes:       {'R', 'G1', 'G2', 'B'} -> (H/2, W/2), black removed
    :param pattern:      raw_pattern, the 2x2 array of colour indices
    :param black_levels: {name: pedestal} to add back
    :return:             (mosaic, cfa_pattern) with cfa_pattern in DNG order
    """
    pattern = np.asarray(pattern)
    height, width = planes['R'].shape
    mosaic = np.zeros((height * 2, width * 2), dtype=np.float64)

    # colour_desc is 'RGBG', so index 1 and 3 are the two greens
    green_seen = 0
    names = {}
    for index in (0, 1, 2, 3):
        letter = 'RGBG'[index]
        if letter == 'G':
            green_seen += 1
            names[index] = f'G{green_seen}'
        else:
            names[index] = letter

    cfa = np.zeros((2, 2), dtype=np.uint8)
    for dy in range(2):
        for dx in range(2):
            colour_index = int(pattern[dy, dx])
            name = names[colour_index]
            mosaic[dy::2, dx::2] = (planes[name]
                                    + black_levels[name])
            # DNG CFAPattern uses 0=red, 1=green, 2=blue
            cfa[dy, dx] = {'R': 0, 'G': 1, 'B': 2}['RGBG'[colour_index]]

    ceiling = white_level if white_level is not None else np.iinfo(dtype).max
    mosaic = np.clip(np.nan_to_num(mosaic, nan=0.0), 0, ceiling)
    return np.rint(mosaic).astype(dtype), cfa.ravel()


def write_bayer_dng(path, mosaic, cfa_pattern, black_level, white_level,
                    colour_matrix=None, white_balance=None,
                    camera_model='astrostack', software='astrostack',
                    datetime_original=None, original_name=None):
    """Write a genuine CFA DNG -- structurally what the camera produced.

    Unlike the linear DNG this carries the mosaic itself, so the reader
    demosaics it with its own algorithm and the values stay in native ADU
    with the camera's own black and white levels. No rescaling, no
    reinterpretation.
    """
    data = np.asarray(mosaic, dtype='<u2')
    if data.ndim != 2:
        raise ValueError(f'expected a 2-D mosaic, got {data.shape}')
    height, width = data.shape

    tiff = _Tiff()
    tiff.add(NEW_SUBFILE_TYPE, LONG, 0)
    tiff.add(IMAGE_WIDTH, LONG, width)
    tiff.add(IMAGE_LENGTH, LONG, height)
    tiff.add(BITS_PER_SAMPLE, SHORT, [16])
    tiff.add(COMPRESSION, SHORT, 1)
    tiff.add(PHOTOMETRIC, SHORT, 32803)               # ColorFilterArray
    tiff.add(SAMPLES_PER_PIXEL, SHORT, 1)
    tiff.add(ROWS_PER_STRIP, LONG, height)
    tiff.add(STRIP_BYTE_COUNTS, LONG, data.nbytes)
    tiff.add(STRIP_OFFSETS, LONG, 0)
    tiff.add(PLANAR_CONFIG, SHORT, 1)
    tiff.add(SAMPLE_FORMAT, SHORT, [1])
    tiff.add(SOFTWARE, ASCII, software)
    if datetime_original:
        tiff.add(DATETIME, ASCII, datetime_original)
    if original_name:
        tiff.add(ORIGINAL_RAW_FILE_NAME, ASCII, original_name)

    tiff.add(CFA_REPEAT_DIM, SHORT, [2, 2])
    tiff.add(CFA_PATTERN, BYTE, [int(v) for v in cfa_pattern])
    tiff.add(CFA_PLANE_COLOR, BYTE, [0, 1, 2])        # red, green, blue
    tiff.add(CFA_LAYOUT, SHORT, 1)                    # rectangular

    tiff.add(DNG_VERSION, BYTE, [1, 4, 0, 0])
    tiff.add(DNG_BACKWARD, BYTE, [1, 1, 0, 0])
    tiff.add(UNIQUE_CAMERA_MODEL, ASCII, camera_model)
    tiff.add(BLACK_LEVEL, SHORT, [int(round(black_level))])
    tiff.add(WHITE_LEVEL, LONG, [int(white_level)])
    tiff.add(DEFAULT_SCALE, RATIONAL, [_rational(1.0), _rational(1.0)])

    if colour_matrix is not None:
        matrix = np.asarray(colour_matrix, dtype=float).reshape(3, 3)
        tiff.add(COLOR_MATRIX_1, SRATIONAL,
                 [_rational(v) for v in matrix.ravel()])
        tiff.add(CALIBRATION_ILLUMINANT_1, SHORT, 21)
    if white_balance is not None:
        multipliers = np.asarray(white_balance, dtype=float)[:3]
        multipliers = np.where(multipliers > 0, multipliers, 1.0)
        neutral = multipliers[1] / multipliers
        tiff.add(AS_SHOT_NEUTRAL, RATIONAL,
                 [_rational(v, 100000) for v in neutral])

    with open(path, 'wb') as handle:
        handle.write(tiff.build(data.tobytes()))
    return path