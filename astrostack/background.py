"""Sky background estimation and bad-pixel characterisation."""

import warnings

import numpy as np
import scipy as sp


def robust_sigma(values) -> float:
    """Gaussian-equivalent spread from the median absolute deviation.

    For X ~ N(μ, σ^2) the median of |X - μ| is σ * Φ^-1(0.75),
    so dividing by 0.6745---multiplying by 1.4826---recovers σ. Unlike
    np.std this ignores the tails, so it measures the noise core rather than
    the RTS pixels and stars sitting on top of it.
    """
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    centre = np.median(values)
    return float(1.4826 * np.median(np.abs(values - centre)))


def robust_σ(values) -> float:
    return robust_sigma(values)


def compute_background(img, box=64, median_filter=3, clip=3, iterations=3, stride=1):
    """σ-clipped mesh background. Accepts (H, W) or (N, H, W).

    A single global threshold is invalid across a light-pollution gradient,
    so the sky is estimated on a coarse mesh and splined back up. `box` must
    be much larger than a star (so no tile is dominated by one) and much
    smaller than the gradient's scale length (so the gradient is tracked).

    :param stride: decimate pixels within each tile. 3 keeps ~455 of 4096
                   samples, runs 2.3x faster, costs ~0.2 ADU of accuracy.
    """
    array = np.asarray(img, dtype=np.float32)
    single = array.ndim == 2
    if single: array = array[None]
    n_frames, height, width = array.shape

    ny, nx = height // box, width // box
    if ny < 3 or nx < 3: raise ValueError(f'box={box} leaves only {ny}x{nx} tiles; '
                                          f'the mesh median filter needs neighbours')

    tiles = array[:, :ny * box, :nx * box].reshape(n_frames, ny, box, nx, box)
    tiles = tiles.transpose(0, 1, 3, 2, 4)
    if stride > 1: tiles = tiles[:, :, :, ::stride, ::stride]
    tiles = np.ascontiguousarray(tiles).reshape(n_frames, ny, nx, -1)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        medians = np.nanmedian(tiles, axis=3)
        for _ in range(iterations):
            deviation = np.abs(tiles - medians[..., None])
            spread = 1.4826 * np.nanmedian(deviation, axis=3)
            ok = deviation < clip * spread[..., None]
            clipped = np.nanmedian(np.where(ok, tiles, np.nan), axis=3)
            # ok.any() guards two cases that both empty a tile: an all-NaN
            # tile, and a saturated one where every pixel is identical so
            # the spread is exactly zero.
            medians = np.where(ok.any(axis=3), clipped, medians)

    # a star bright enough to dominate a whole tile survives clipping; the
    # mesh median lets its eight neighbours outvote it
    medians = sp.ndimage.median_filter(medians, size=(1, median_filter, median_filter), mode='nearest')

    for i in range(n_frames):
        bad = ~np.isfinite(medians[i])
        if bad.all(): medians[i] = 0.0
        elif bad.any():
            nearest = sp.ndimage.distance_transform_edt(bad, return_distances=False, return_indices=True)
            medians[i] = medians[i][nearest[0], nearest[1]]

    # tile i holds the median of the pixels it sampled, so its value belongs
    # at their centroid -- not at output row i. linspace(0, ny-1, height)
    # would stretch the mesh and misplace it by a whole tile at the far edge.
    offset = float(np.arange(0, box, stride).mean())
    zy = (np.arange(height) - offset) / box
    zx = (np.arange(width) - offset) / box
    grid = np.meshgrid(zy, zx, indexing='ij')

    out = np.empty((n_frames, height, width), dtype=np.float32)
    for i in range(n_frames):
        out[i] = sp.ndimage.map_coordinates(medians[i], grid, order=3, mode='nearest')
    return out[0] if single else out


def residuals(cube, background=None, box=64):
    """Background-subtracted cube plus each frame's own level and noise.

    One background serves the whole cube: the sky pattern is stable across a
    session, and using a single estimate makes any residual structure
    identical in every frame, so it cancels in the differences registration
    measures. The per-frame `level` then absorbs sky drift.

    :return: (residual, level, σ) with level and σ shaped (N, 1, 1)
             so they broadcast straight back against the cube
    """
    if background is None: background = compute_background(cube[len(cube) // 2], box=box)
    residual = cube - background
    level = np.nanmedian(residual, axis=(1, 2), keepdims=True)
    σ = 1.4826 * np.nanmedian(np.abs(residual - level), axis=(1, 2), keepdims=True)
    return residual, level, σ


def cold_pixel_map(cube, threshold=-50.0, min_frames=3):
    """Pixels that read far below the pedestal in several frames.

    Random telegraph signal: a fixed set of photosites whose readout
    transistor flickers between states. Fixed *locations*, varying *state*,
    which is why a master dark cannot remove them---subtracting the mean
    leaves the flicker. Sigma-clipped stacking is what handles them.

    :return: (mask, counts)---counts[i, j] is how many frames pixel (i, j)
             went cold in, which is the diagnostic worth plotting
    """
    counts = (cube < threshold).sum(axis=0)
    return counts >= min_frames, counts


def fill_nonfinite(cube, inplace=False):
    """Replace every NaN with its own frame's median.

    Detection cannot see NaN: a single one inside a star's measurement
    window makes the moments NaN, so the centroid is NaN, so the triangle
    invariants are NaN, and the KDTree in the matcher rejects the whole
    frame. Substituting the sky level is right for detection---these are
    isolated rejected samples, and the value only decides which stars are
    found, never what is measured. Stacking keeps the original NaN cube.
    """
    out = cube if inplace else cube.copy()
    for i, frame in enumerate(out):
        bad = ~np.isfinite(frame)
        if bad.any(): frame[bad] = np.nanmedian(frame)
    return out


def fill_bad_pixels(cube, bad_pixels):
    """Nearest-neighbour fill, for detection only.

    Detection filters have wide footprints and NaN propagates through all of
    them, so one bad pixel would wipe out a 33x33 neighbourhood. Stacking
    still uses the original cube, so filled values never enter a measurement
    ---they only influence which stars are found.
    """
    if bad_pixels is None or not bad_pixels.any(): return cube
    nearest = sp.ndimage.distance_transform_edt(bad_pixels, return_distances=False,
                                                return_indices=True)
    return cube[:, nearest[0], nearest[1]]