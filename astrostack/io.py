"""Reading raw files into linear, black-subtracted Bayer planes."""

import numpy as np


def load_planes(path: str) -> dict[str, np.ndarray]:
    """Load a raw file and split the Bayer mosaic into its four sub-lattices.

    The mosaic is *not* demosaiced. Demosaicing interpolates between
    neighbouring photosites, which correlates the noise and destroys the
    per-pixel independence every later noise estimate relies on. Splitting is
    lossless: 4 planes of (H/2, W/2) hold exactly the H*W original samples.

    :param path: path to a raw file (CR2, NEF, ...)
    :return:     {'R', 'G1', 'G2', 'B'} -> (H/2, W/2) float32, black subtracted
    """
    import rawpy

    with rawpy.imread(path) as raw:
        bayer = raw.raw_image_visible.astype(np.float32)
        pattern = raw.raw_pattern
        black = np.asarray(raw.black_level_per_channel, np.float32)

    height, width = (size - size % 2 for size in bayer.shape)
    planes, green = {}, 0
    for dy in range(2):
        for dx in range(2):
            colour_index = int(pattern[dy, dx])
            name = 'RGBG'[colour_index]
            if name == 'G':
                green += 1
                name = f'G{green}'
            # after black subtraction about half the sky pixels are negative.
            # that is correct: clipping at zero truncates the lower half of
            # the noise distribution and biases every background estimate.
            planes[name] = bayer[dy:height:2, dx:width:2] - black[colour_index]
    return planes


def load_metadata(path: str) -> dict:
    """Exposure, timing and sensor limits for one raw file.

    `timestamp` is what lets you convert a fitted shift into a drift rate,
    and hence into a pixel scale.
    """
    import rawpy

    with rawpy.imread(path) as raw:
        return dict(white_level=int(raw.white_level),
                    black_level=list(raw.black_level_per_channel),
                    pattern=raw.raw_pattern.tolist(),
                    colour_description=raw.color_desc.decode(),
                    shape=raw.raw_image_visible.shape,
                    camera_white=list(raw.camera_white_level_per_channel)
                    if hasattr(raw, 'camera_white_level_per_channel') else None)


def load_cube(paths, channel='green') -> np.ndarray:
    """Stack many raw files into one (n_frames, H/2, W/2) array.

    :param channel: 'green' averages G1 and G2 -- the most photons and the
                    best sampling, so the right choice for detection and
                    registration. Any of 'R', 'G1', 'G2', 'B' selects one
                    plane instead.

    The generator means only one file's four planes are alive at a time
    rather than all of them.
    """
    def extract(planes):
        if channel == 'green':
            return 0.5 * (planes['G1'] + planes['G2'])
        return planes[channel]

    return np.stack([extract(planes)
                     for planes in (load_planes(path) for path in paths)])


def saturation_level(metadata: dict, fraction: float = 0.90) -> float:
    """Where to call a pixel saturated, in black-subtracted ADU.

    Sensors go non-linear before they hit the hard ceiling, so the usable
    limit sits below `white_level - black_level`, not at it.
    """
    black = float(np.median(metadata['black_level']))
    return fraction * (metadata['white_level'] - black)
