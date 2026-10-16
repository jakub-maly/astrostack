"""Star detection: threshold, label, and measure."""

import numpy as np
import polars as pl
import scipy as sp


def detect_frame(residual, level, σ, threshold=5, min_area=4, edge=26,
                 box=21, saturation=None, bad_pixels=None, raw_frame=None,
                 max_stars=None, dilate=True):
    """Find stars in one background-subtracted frame.

    :param residual: (H, W) frame with the background already removed
    :param level:    that frame's residual median---the threshold is
                     referenced to it so sky drift between frames does not
                     shift the effective cut
    :param σ:        that frame's noise, from robust_sigma
    :param box:      side of the measurement window. Roughly 2.6 x FWHM:
                     too small truncates the PSF and degrades centroids,
                     too large swallows neighbouring stars.
    :return:         dict of per-star arrays, brightest first, or None
    """
    # ---- 1. threshold and clean up --------------------------------------
    mask = residual > level + threshold * σ
    # opening needs a full 2x2 block above threshold, so an isolated hot or
    # RTS pixel cannot survive however extreme its value
    mask = sp.ndimage.binary_opening(mask, np.ones((2, 2)))
    if dilate:
        # the threshold amputated the PSF wings; growing the footprint back
        # takes the captured flux from ~84% to ~99%. it can merge pairs
        # closer than about 3x the FWHM, so turn it off if that matters.
        mask = sp.ndimage.binary_dilation(mask, np.ones((3, 3)))

    labels, n_objects = sp.ndimage.label(mask)
    if n_objects == 0:
        return None

    # ---- 2. one statistic per connected component -----------------------
    index = np.arange(1, n_objects + 1)
    signal = residual - level
    weights = np.clip(signal, 0, None)     # a weighted mean needs w >= 0
    area = np.bincount(labels.ravel(), minlength=n_objects + 1)[1:]
    # flux sums the UNCLIPPED signal: clipping discards downward noise while
    # keeping upward, which biases the total by a brightness-dependent amount
    flux = sp.ndimage.sum_labels(signal, labels, index)
    peak = sp.ndimage.maximum(residual, labels, index)
    centre = np.array(sp.ndimage.center_of_mass(weights, labels, index))

    # ---- 3. cheap cuts before the expensive part ------------------------
    height, width = residual.shape
    keep = ((area >= min_area) & np.isfinite(centre).all(axis=1) &
            (centre[:, 0] > edge) & (centre[:, 0] < height - edge) &
            (centre[:, 1] > edge) & (centre[:, 1] < width - edge))
    if not keep.any():
        return None
    area, flux, peak, centre = area[keep], flux[keep], peak[keep], centre[keep]

    # ---- 4. sub-pixel centroids and shape, all stars at once ------------
    radius = box // 2
    rows = np.round(centre[:, 0]).astype(np.intp)
    columns = np.round(centre[:, 1]).astype(np.intp)
    offset_y, offset_x = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    # (n_stars, box, box) index arrays -> every window in one fancy index.
    # the edge cut above is what keeps these in bounds: negative indices
    # would WRAP to the far side of the frame without raising.
    window_rows = rows[:, None, None] + offset_y[None]
    window_columns = columns[:, None, None] + offset_x[None]

    window = np.clip(residual[window_rows, window_columns] - level, 0, None)
    total = window.sum(axis=(1, 2))
    valid = total > 0
    safe_total = np.where(valid, total, 1.0)

    # first moments: a weighted average of integer positions is continuous
    # in the true position, because the WEIGHTS move even though the
    # coordinates do not. that is where sub-pixel accuracy comes from.
    mean_y = (window * offset_y).sum(axis=(1, 2)) / safe_total
    mean_x = (window * offset_x).sum(axis=(1, 2)) / safe_total
    centre_y, centre_x = rows + mean_y, columns + mean_x

    centred_y = offset_y - mean_y[:, None, None]
    centred_x = offset_x - mean_x[:, None, None]
    var_yy = (window * centred_y ** 2).sum(axis=(1, 2)) / safe_total
    var_xx = (window * centred_x ** 2).sum(axis=(1, 2)) / safe_total
    var_xy = (window * centred_y * centred_x).sum(axis=(1, 2)) / safe_total

    # eigenvalues of [[var_yy, var_xy], [var_xy, var_xx]]. using var_yy and
    # var_xx alone would report zero ellipticity for a 45-degree trail.
    half_trace = 0.5 * (var_yy + var_xx)
    half_difference = 0.5 * np.sqrt((var_yy - var_xx) ** 2 + 4.0 * var_xy ** 2)
    major = np.clip(half_trace + half_difference, 1e-6, None)
    minor = np.clip(half_trace - half_difference, 1e-6, None)

    fwhm = 2.3548 * np.sqrt(np.clip(half_trace, 1e-6, None))
    ellipticity = 1.0 - np.sqrt(minor / major)
    orientation = np.degrees(0.5 * np.arctan2(2.0 * var_xy, var_yy - var_xx))

    # ---- 5. flags -------------------------------------------------------
    if saturation is None:
        saturated = np.zeros(len(centre_y), bool)
    else:
        source = raw_frame if raw_frame is not None else residual + level
        saturated = (source[window_rows, window_columns]
                     > saturation).any(axis=(1, 2))
    if bad_pixels is None:
        on_badpix = np.zeros(len(centre_y), bool)
    else:
        on_badpix = bad_pixels[window_rows, window_columns].any(axis=(1, 2))

    # a NaN centroid would survive all the way to the KDTree in
    # match_asterisms and fail there, three call levels from its cause
    finite = (np.isfinite(centre_y) & np.isfinite(centre_x)
              & np.isfinite(fwhm) & np.isfinite(flux) & valid)
    if not finite.any():
        return None

    out = dict(y=centre_y, x=centre_x, flux=flux, peak=peak,
               area=area.astype(np.int32), fwhm=fwhm,
               ellipticity=ellipticity, orientation=orientation,
               snr=flux / (σ * np.sqrt(area)),
               saturated=saturated, on_badpix=on_badpix)

    # integrated flux, not peak: saturated stars share an identical clipped
    # peak, so a peak sort orders them arbitrarily and differently per frame
    out = {key: value[finite] for key, value in out.items()}
    order = np.argsort(-out['flux'])[:max_stars]
    return {key: value[order] for key, value in out.items()}


def detect_cube(residual, level, sigma, frame_index=None, **kwargs):
    """Run detect_frame over every frame and return one long-format table.

    The loop is not laziness: sp.ndimage.label on a 3-D array treats the
    frame axis as spatial, so the same star in consecutive frames would
    merge into a single component with one centroid.
    """
    if frame_index is None:
        frame_index = np.arange(len(residual))

    columns, frames = [], []
    for position, label in enumerate(frame_index):
        found = detect_frame(residual[position],
                             float(np.ravel(level[position])[0]),
                             float(np.ravel(sigma[position])[0]), **kwargs)
        if found is None:
            continue
        columns.append(found)
        # the true label, not the position: if one frame yields nothing,
        # using the position would mislabel every frame after it
        frames.append(np.full(len(found['y']), int(label), dtype=np.int32))

    if not columns:
        return None
    table = {key: np.concatenate([entry[key] for entry in columns])
             for key in columns[0]}
    table['frame'] = np.concatenate(frames)
    return pl.DataFrame(table)


def select_stars(detections, frame, max_stars=25, exclude_saturated=True,
                 exclude_badpix=False, max_ellipticity=None,
                 min_keep=8):
    """Positions of one frame's best stars, as (n_stars, 2) of (y, x).

    Saturated stars have flat cores, so they are dropped by default.

    `on_badpix` is far less damaging and is kept by default: the flag fires
    if *any* pixel in the whole measurement window touches the bad-pixel
    map, and with a 21-pixel box that is 441 pixels per star, so on a sensor
    with a percent or two of RTS pixels most stars trip it. A single bad
    sample among 441 barely moves a flux-weighted centroid. Set
    `exclude_badpix=True` if your defect population is severe.

    Every cut is skipped rather than applied when it would leave fewer than
    `min_keep` stars -- a poor star beats no star.
    """
    rows = detections.filter(pl.col('frame') == frame)
    for column, active in (('saturated', exclude_saturated),
                           ('on_badpix', exclude_badpix)):
        if active and column in rows.columns:
            kept = rows.filter(~pl.col(column))
            if len(kept) >= min_keep:
                rows = kept
    if max_ellipticity is not None:
        rounder = rows.filter(pl.col('ellipticity') <= max_ellipticity)
        if len(rounder) >= min_keep:
            rows = rounder
    rows = rows.sort('flux', descending=True).head(max_stars)
    positions = rows.select(['y', 'x']).to_numpy()
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError(f'expected (n, 2), got {positions.shape}')
    positions = positions[np.isfinite(positions).all(axis=1)]
    if len(positions) < 3:
        raise ValueError(f'frame {frame}: only {len(positions)} usable star '
                         'positions after filtering')
    return positions