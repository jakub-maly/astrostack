"""Reading camera EXIF straight out of a raw file.

CR2, NEF, ARW and DNG are all TIFF containers, so the standard EXIF image
file directories can be walked without rawpy (which does not expose them)
or an external exiftool. Only the handful of tags an astronomical header
wants are decoded.
"""

import struct

# tag -> (name, which directory it lives in)
IFD0_TAGS = {0x010F: 'make', 0x0110: 'model', 0x0132: 'datetime',
             0x8769: '_exif_pointer'}
EXIF_TAGS = {0x829A: 'exposure_time', 0x829D: 'f_number',
             0x8827: 'iso', 0x9003: 'datetime_original',
             0x920A: 'focal_length', 0xA434: 'lens_model',
             0x9204: 'exposure_bias', 0xA405: 'focal_length_35mm',
             0x9291: 'subsec_original'}

# EXIF type code -> (struct format, bytes per component)
TYPES = {1: ('B', 1), 2: ('s', 1), 3: ('H', 2), 4: ('I', 4), 5: ('II', 8),
         6: ('b', 1), 7: ('B', 1), 8: ('h', 2), 9: ('i', 4), 10: ('ii', 8),
         11: ('f', 4), 12: ('d', 8)}


def _read_value(data, byte_order, type_code, count, offset_field, base=0):
    if type_code not in TYPES:
        return None
    fmt, size = TYPES[type_code]
    total = size * count
    if total <= 4:
        payload = offset_field
    else:
        start = base + struct.unpack(byte_order + 'I', offset_field)[0]
        payload = data[start:start + total]
    if type_code == 2:
        return payload.split(b'\x00')[0].decode('ascii', 'replace').strip()
    if type_code in (5, 10):
        values = struct.unpack(byte_order + ('II' if type_code == 5 else 'ii')
                               * count, payload[:total])
        pairs = [(values[i], values[i + 1]) for i in range(0, len(values), 2)]
        results = [n / d if d else 0.0 for n, d in pairs]
        return results[0] if count == 1 else results
    values = struct.unpack(byte_order + fmt * count, payload[:total])
    return values[0] if count == 1 else list(values)


def _walk_ifd(data, byte_order, offset, tags, base=0):
    found = {}
    if offset + 2 > len(data):
        return found
    count, = struct.unpack(byte_order + 'H', data[offset:offset + 2])
    for i in range(count):
        entry = offset + 2 + i * 12
        if entry + 12 > len(data):
            break
        tag, type_code, n = struct.unpack(byte_order + 'HHI',
                                          data[entry:entry + 8])
        if tag not in tags:
            continue
        found[tags[tag]] = _read_value(data, byte_order, type_code, n,
                                       data[entry + 8:entry + 12], base)
    return found


def read_exif(path, max_bytes=2_000_000):
    """Camera settings from a TIFF-based raw file.

    Only the header region is read, not the image data, so this is fast even
    on a 24 MB file.

    :return: dict with whichever of make, model, lens_model, iso,
             exposure_time, f_number, focal_length and datetime_original
             the file carries. Missing keys simply mean the tag is absent.
    """
    with open(path, 'rb') as handle:
        data = handle.read(max_bytes)
    if len(data) < 8:
        return {}

    marker = data[:2]
    if marker == b'II':
        byte_order = '<'
    elif marker == b'MM':
        byte_order = '>'
    else:
        return {}

    first_ifd, = struct.unpack(byte_order + 'I', data[4:8])
    info = _walk_ifd(data, byte_order, first_ifd, IFD0_TAGS)
    pointer = info.pop('_exif_pointer', None)
    if pointer:
        info.update(_walk_ifd(data, byte_order, int(pointer), EXIF_TAGS))
    return info


def exif_to_fits_header(exif, extra=None):
    """Translate EXIF into the FITS keywords astronomical tools expect.

    DATE-OBS, EXPTIME and INSTRUME are the standard names; the rest follow
    the conventions used by amateur astronomy software.
    """
    header = {}
    if exif.get('datetime_original'):
        # EXIF writes 'YYYY:MM:DD HH:MM:SS'; FITS wants ISO 8601
        stamp = exif['datetime_original']
        date, _, clock = stamp.partition(' ')
        iso = f"{date.replace(':', '-')}T{clock}"
        if exif.get('subsec_original'):
            iso = f"{iso}.{str(exif['subsec_original']).strip()}"
        header['DATE-OBS'] = (iso, 'start of the reference exposure, local')
    if exif.get('exposure_time'):
        header['EXPTIME'] = (float(exif['exposure_time']),
                             'single sub-exposure, seconds')
    if exif.get('model'):
        model = exif['model']
        maker = (exif.get('make') or '').split()[0] if exif.get('make') else ''
        # Canon writes 'Canon EOS 70D' as the model AND 'Canon' as the make
        name = model if maker and maker.lower() in model.lower() \
            else f'{maker} {model}'.strip()
        header['INSTRUME'] = (name, 'camera')
    if exif.get('lens_model'):
        header['TELESCOP'] = (exif['lens_model'], 'lens or telescope')
    if exif.get('focal_length'):
        header['FOCALLEN'] = (float(exif['focal_length']),
                              'focal length, mm')
    if exif.get('f_number'):
        header['APTDIA'] = (round(float(exif['focal_length'])
                                  / float(exif['f_number']), 2),
                            'aperture diameter, mm') \
            if exif.get('focal_length') else None
        header['FNUMBER'] = (float(exif['f_number']), 'focal ratio')
    if exif.get('iso'):
        header['ISOSPEED'] = (int(exif['iso']), 'camera ISO setting')
    header = {k: v for k, v in header.items() if v is not None}
    header.update(extra or {})
    return header


# pixel pitch is not in EXIF, so the common bodies are tabulated. Add yours
# if it is missing; the plate scale is only used for a header comment.
SENSOR_PITCH_UM = {
    'canon eos 70d': 4.11, 'canon eos 80d': 3.72, 'canon eos 7d mark ii': 4.11,
    'canon eos 6d': 6.55, 'canon eos 5d mark iii': 6.25,
    'canon eos 60d': 4.30, 'canon eos 600d': 4.30, 'canon eos 700d': 4.30,
    'canon eos 750d': 3.72, 'canon eos 77d': 3.72, 'canon eos r': 5.36,
    'nikon d5300': 3.92, 'nikon d750': 5.97, 'sony ilce-6000': 3.92,
}


def sensor_pitch(exif):
    """Pixel pitch in microns for a known body, else None."""
    model = (exif.get('model') or '').strip().lower()
    return SENSOR_PITCH_UM.get(model)


def pixel_scale(focal_length_mm, pixel_size_um, binning=2):
    """Plate scale in arcsec per pixel.

    `binning` is 2 for superpixel planes, where one output pixel spans a 2x2
    Bayer tile. The sensor's pixel size is not in EXIF and has to come from
    the camera's specifications.
    """
    return 206265.0 * (pixel_size_um * 1e-3) * binning / focal_length_mm
