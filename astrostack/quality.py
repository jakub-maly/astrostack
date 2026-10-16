import numpy as np
import polars as pl


def frame_quality(detections, frame_index=None):
    """Per-frame summary of the detection table.

    `frame_index` is the full list of frame labels in the cube. Pass it when
    a frame might have produced no detections at all -- that frame is
    invisible in the table, so it cannot be recovered from the data, and
    omitting it here would silently shift every later index. When it is not
    given, the index is taken from the labels actually present, which copes
    with 1-indexed or non-contiguous labelling but not with empty frames.
    """
    label_dtype = detections['frame'].dtype
    if frame_index is None:
        frame_index = np.sort(detections['frame'].unique().to_numpy())
    else:
        frame_index = np.asarray(list(frame_index))
        observed = detections['frame'].unique().to_numpy()
        missing = np.setdiff1d(observed, frame_index)
        if len(missing):
            raise ValueError(f'detections reference frames {missing} that are '
                             f'not in frame_index')

    summary = (detections
               .group_by('frame')
               .agg(n_stars=pl.len(),
                    fwhm=pl.col('fwhm').median(),
                    ellipticity=pl.col('ellipticity').median()
                    if 'ellipticity' in detections.columns else pl.lit(None),
                    snr=pl.col('snr').median(),
                    flux=pl.col('flux').sum(),
                    saturated=pl.col('saturated').sum()))

    # the join key must match in dtype as well as value: detect_cube writes
    # `frame` as Int32 while np.arange gives Int64, and a mismatched left
    # join returns nulls for every row rather than raising, which silently
    # turns every frame into "too few stars"
    index = (pl.DataFrame({'frame': frame_index})
             .with_columns(frame=pl.col('frame').cast(label_dtype)))
    joined = index.join(summary, on='frame', how='left')
    if len(detections) and joined['n_stars'].null_count() == len(joined):
        raise RuntimeError(
            f'no frame in frame_index matched the detection table. '
            f'Labels present: {sorted(detections["frame"].unique().to_list())[:8]}; '
            f'index given: {list(frame_index)[:8]}')
    return joined.with_columns(n_stars=pl.col('n_stars').fill_null(0)).sort('frame')


def usable_frames(detections, frame_index=None, min_stars=8,
                  max_fwhm_ratio=2.0, max_ellipticity=0.6,
                  min_flux_ratio=0.4):
    """Frame indices fit to stack, plus the quality table with the verdict.

    The cuts target frames that are *broken*, not merely soft. A soft frame
    still delivers photons and discarding it costs sqrt(N) unconditionally,
    so the thresholds are deliberately loose: they catch cloud, a bumped
    tripod, a pointing error or a tracking failure, and let everything else
    through to be inverse-variance weighted at the stacking stage.

    Returns (frames, quality) where `frames` is an int array and `quality`
    carries a boolean `usable` column and a `reason` for each rejection.
    """
    quality = frame_quality(detections, frame_index)
    populated = quality.filter(pl.col('n_stars') >= min_stars)
    if len(populated) == 0:
        raise RuntimeError('no frame has enough detections to assess')

    median_fwhm = float(populated['fwhm'].median())
    median_flux = float(populated['flux'].median())

    quality = quality.with_columns(
        fwhm_ratio=pl.col('fwhm') / median_fwhm,
        flux_ratio=pl.col('flux') / median_flux)

    too_few = pl.col('n_stars') < min_stars
    too_soft = pl.col('fwhm_ratio') > max_fwhm_ratio
    too_elongated = (pl.col('ellipticity') > max_ellipticity
                     if 'ellipticity' in quality.columns else pl.lit(False))
    too_faint = pl.col('flux_ratio') < min_flux_ratio

    quality = quality.with_columns(
        reason=pl.when(too_few).then(pl.lit('too few stars'))
                 .when(too_faint).then(pl.lit('lost transparency'))
                 .when(too_soft).then(pl.lit('badly defocused or trailed'))
                 .when(too_elongated).then(pl.lit('elongated: wind or shake'))
                 .otherwise(pl.lit(None)))
    quality = quality.with_columns(usable=pl.col('reason').is_null())

    frames = quality.filter(pl.col('usable'))['frame'].to_numpy()
    if len(frames) == 0:
        tally = {}
        for reason in quality['reason'].to_list():
            tally[reason] = tally.get(reason, 0) + 1
        detail = ', '.join(f'{count} {reason}'
                           for reason, count in sorted(tally.items(),
                                                       key=lambda kv: -kv[1]))
        raise RuntimeError(
            f'every frame was rejected ({detail}). '
            f'Median FWHM {median_fwhm:.2f} px, median total flux '
            f'{median_flux:.3g}. Loosen the thresholds via min_stars, '
            f'max_fwhm_ratio, max_ellipticity or min_flux_ratio.')
    return frames, quality


def reference_frame(quality):
    """Sharpest usable frame. It is the only one never resampled, so it
    should be the best one you have."""
    usable = quality.filter(pl.col('usable'))
    return int(usable.sort('fwhm')['frame'][0])


def keep_frames(detections, frames):
    """Detection table restricted to the given frames."""
    return detections.filter(pl.col('frame').is_in(list(map(int, frames))))