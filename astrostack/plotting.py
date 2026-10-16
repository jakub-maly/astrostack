"""Diagnostics for every stage of the pipeline.

Everything here returns a matplotlib figure. Altair renders marks rather
than rasters, so a five-megapixel frame cannot be a Vega layer; Altair is
used for the tabular diagnostics instead, where its interactivity earns its
place.
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

ACCENT = '#1F5C8B'
WARN = '#C4622D'


# ----------------------------------------------------------------- display

def stretch(img, lo_pct=1, target_bg=0.20):
    """Midtone transfer function.

    Linear astronomical data is unviewable: the sky sits a few ADU above
    zero and the stars saturate the top end, so a linear map shows black
    with a few white dots. This puts the sky background at `target_bg` and
    leaves the highlights intact.
    """
    finite = img[np.isfinite(img)]
    low = np.percentile(finite, lo_pct)
    high = finite.max()
    middle = np.median(finite)
    scaled = np.clip((img - low) / max(high - low, 1e-9), 0, 1)
    m = (middle - low) / max(high - low, 1e-9)
    b = (target_bg - 1) * m / ((2 * target_bg - 1) * m - target_bg)
    b = float(np.clip(b, 1e-4, 1 - 1e-4))
    return ((b - 1) * scaled) / ((2 * b - 1) * scaled - b)


def show(ax, img, decimate=None, **kwargs):
    """imshow with the astro conventions applied.

    Matplotlib's own downsampling drops stars and displaces those it keeps
    by a pixel or two, so large frames are decimated explicitly and the
    extent is set to keep the axes in full-resolution coordinates.
    """
    if decimate is None:
        decimate = max(1, int(np.ceil(max(img.shape[1] / 1400,
                                          img.shape[0] / 1000))))
    ax.imshow(stretch(img[::decimate, ::decimate]), cmap='gray',
              origin='upper', interpolation='nearest',
              extent=[-0.5, img.shape[1] - 0.5, img.shape[0] - 0.5, -0.5],
              **kwargs)
    ax.set_xticks([])
    ax.set_yticks([])
    return ax


# ------------------------------------------------------------------ step 1

def plot_planes(planes, size=(11, 3)):
    """The four Bayer sub-lattices side by side.

    Splitting is a re-indexing, not a reduction: 4 planes of (H/2, W/2)
    hold exactly the H*W original samples.
    """
    order = ['R', 'G1', 'G2', 'B']
    fig, axes = plt.subplots(1, 4, figsize=size)
    for ax, name in zip(axes, order):
        show(ax, planes[name])
        ax.set_title(f'{name}   median {np.median(planes[name]):.0f} ADU',
                     fontsize=9)
    fig.tight_layout()
    return fig


def plot_pixel_histogram(cube, frame=0, sigma=None, threshold=None,
                         cold_cut=-50, size=(6, 3.5)):
    """Where the sky core, the cold tail and the stars sit.

    Log counts, because the three populations differ by orders of magnitude.
    A single Gaussian would give one straight parabola; a second population
    shows as a shoulder.
    """
    values = cube[frame][np.isfinite(cube[frame])]
    fig, ax = plt.subplots(figsize=size)
    ax.hist(values, bins=400, range=np.percentile(values, [0.01, 99.9]),
            color=ACCENT, histtype='stepfilled', lw=0)
    ax.set_yscale('log')
    ax.set_xlabel('ADU above black level')
    ax.set_ylabel('pixel count')
    if cold_cut is not None:
        ax.axvline(cold_cut, color=WARN, ls='--', lw=1,
                   label=f'cold cut {cold_cut}')
    if sigma is not None:
        level = np.median(values)
        ax.axvspan(level - sigma, level + sigma, color=ACCENT, alpha=0.15,
                   label=f'$\\pm\\sigma$ = {sigma:.1f} ADU')
        if threshold is not None:
            ax.axvline(level + threshold * sigma, color='k', ls=':', lw=1,
                       label=f'{threshold:.0f}$\\sigma$ detection threshold')
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    return fig


def plot_cold_recurrence(counts, size=(6, 3.5)):
    """How many frames each cold pixel is cold in, against independence.

    If cold pixels were random noise the histogram would follow a binomial.
    A huge excess at k >= 3 means a fixed population that flickers -- random
    telegraph signal, which no master dark can remove.
    """
    from math import comb
    n_frames = int(counts.max())
    if n_frames < 2:
        raise ValueError('need at least 2 frames with cold pixels to compare '
                         'against a binomial')
    observed = np.bincount(counts.ravel(), minlength=n_frames + 1)
    total = observed.sum()
    detections = (np.arange(len(observed)) * observed).sum()
    probability = detections / (n_frames * total) if total else 0
    expected = np.array([total * comb(n_frames, k) * probability ** k
                         * (1 - probability) ** (n_frames - k)
                         for k in range(len(observed))])

    fig, ax = plt.subplots(figsize=size)
    k = np.arange(len(observed))
    ax.bar(k, np.maximum(observed, 0.1), color=ACCENT, label='observed')
    ax.plot(k, np.maximum(expected, 0.1), 'o--', color=WARN, ms=4, lw=1,
            label='if cold pixels were independent')
    ax.set_yscale('log')
    ax.set_xlabel('frames in which the pixel reads cold')
    ax.set_ylabel('pixels')
    ax.set_ylim(0.5, None)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------ step 2

def plot_background(frame, background, size=(12, 3.6)):
    """Raw frame, fitted sky, and the residual that detection works on."""
    residual = frame - background
    fig, axes = plt.subplots(1, 3, figsize=size)
    show(axes[0], frame)
    axes[0].set_title('raw plane (light-pollution gradient)', fontsize=9)
    axes[1].imshow(background[::4, ::4], cmap='Greys_r', origin='upper')
    axes[1].set_title(f'mesh background  '
                      f'{background.min():.0f}-{background.max():.0f} ADU',
                      fontsize=9)
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    show(axes[2], residual)
    sigma = 1.4826 * np.median(np.abs(residual - np.median(residual)))
    axes[2].set_title(f'residual   $\\sigma$ = {sigma:.1f} ADU', fontsize=9)
    fig.tight_layout()
    return fig


def plot_detection_stages(residual, level, sigma, threshold=5,
                          window=None, size=(12, 3.6)):
    """Threshold, opening, dilation -- what each morphology step removes.

    Opening needs a full 2x2 block above threshold, so isolated hot and RTS
    pixels cannot survive it. The component count between panels 1 and 2 is
    the number of single-pixel defects it discarded.
    """
    if window is None:
        # centre on the busiest region so the panels show several stars
        span = 300
        coarse = (residual > level + threshold * sigma).astype(np.float32)
        density = sp.ndimage.uniform_filter(coarse, size=span // 2)
        margin = span // 2 + 5
        density[:margin] = density[-margin:] = 0
        density[:, :margin] = density[:, -margin:] = 0
        row, column = np.unravel_index(np.argmax(density), density.shape)
        row, column = int(row) - span // 2, int(column) - span // 2
        window = (row, column, span)
    row, column, span = window
    cut = residual[row:row + span, column:column + span]
    raw = cut > level + threshold * sigma
    opened = sp.ndimage.binary_opening(raw, np.ones((2, 2)))
    dilated = sp.ndimage.binary_dilation(opened, np.ones((3, 3)))

    fig, axes = plt.subplots(1, 4, figsize=size)
    show(axes[0], cut, decimate=1)
    axes[0].set_title('residual', fontsize=9)
    for ax, mask, name in zip(axes[1:], [raw, opened, dilated],
                              ['thresholded', 'opened (2x2)', 'dilated (3x3)']):
        ax.imshow(mask, cmap='gray', origin='upper', interpolation='nearest')
        ax.set_title(f'{name}\n{sp.ndimage.label(mask)[1]} components',
                     fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    return fig


def plot_frame(cube, detections, frame=0, centroids=False, n=9, box=31,
               cols=None, panel=None, max_show=None, drop_flagged=None,
               size=(11, 7.5), title=None):
    """One frame with its detections, or a grid of zoomed centroids.

    The zoom grid is the diagnostic that catches real bugs: every red cross
    should sit visually dead centre. A consistent offset means a coordinate
    convention error; scattered offsets mean blends or cosmic rays.
    """
    import polars as pl
    img = cube[frame] if np.ndim(cube) == 3 else cube
    rows = detections
    if 'frame' in rows.columns:
        rows = rows.filter(pl.col('frame') == frame)
    if drop_flagged is None:
        drop_flagged = centroids
    if drop_flagged:
        for column in ('saturated', 'on_badpix'):
            if column in rows.columns:
                rows = rows.filter(~pl.col(column))
    if len(rows) == 0:
        raise ValueError(f'no detections for frame {frame}')

    data = {c: rows[c].to_numpy() for c in rows.columns}
    order = np.argsort(-data['flux'])
    if not centroids:
        return _overview(img, data, order, max_show, size, title, frame)
    return _zoom(img, data, order[:n] if n else order, box, cols, panel,
                 title, frame)


def _overview(img, data, order, max_show, size, title, frame):
    if max_show is not None:
        order = order[:max_show]
    y, x, flux = data['y'][order], data['x'][order], data['flux'][order]
    n_all = len(data['y'])
    saturated = data.get('saturated', np.zeros(n_all, bool))[order]
    on_badpix = data.get('on_badpix', np.zeros(n_all, bool))[order]

    log_flux = np.log10(np.clip(flux, 1, None))
    marker = 40 + 260 * (log_flux - log_flux.min()) / max(
        log_flux.max() - log_flux.min(), 1e-9)

    fig, ax = plt.subplots(figsize=size)
    show(ax, img)
    for mask, colour, label in [(~saturated & ~on_badpix, '#3DDC84', 'clean'),
                                (saturated & ~on_badpix, '#FFA000', 'saturated'),
                                (on_badpix, '#FF4444', 'bad pixel')]:
        if mask.any():
            ax.scatter(x[mask], y[mask], s=marker[mask], facecolors='none',
                       edgecolors=colour, linewidths=1,
                       label=f'{label} ({mask.sum()})')
    ax.legend(loc='upper right', fontsize=8, framealpha=0.75)
    ax.set_title(title or f'frame {frame} -- {len(y)} detections', fontsize=10)
    return fig


def _zoom(img, data, order, box, cols, panel, title, frame):
    count = len(order)
    if cols is None:
        cols = int(np.ceil(np.sqrt(count * 1.5)))
    rows_n = int(np.ceil(count / cols))
    if panel is None:
        panel = float(np.clip(14 / cols, 0.75, 2.1))
    fontsize = float(np.clip(2.6 * panel, 4.5, 8))

    fig, axes = plt.subplots(rows_n, cols, squeeze=False,
                             figsize=(panel * cols, panel * rows_n))
    radius = box // 2
    height, width = img.shape
    for ax, k in zip(np.ravel(axes), order):
        cy, cx = data['y'][k], data['x'][k]
        yi, xi = int(round(cy)), int(round(cx))
        y0, y1 = max(0, yi - radius), min(height, yi + radius + 1)
        x0, x1 = max(0, xi - radius), min(width, xi + radius + 1)
        ax.imshow(stretch(img[y0:y1, x0:x1], lo_pct=2), cmap='gray',
                  origin='upper', interpolation='nearest',
                  extent=[x0 - 0.5, x1 - 0.5, y1 - 0.5, y0 - 0.5])
        ax.plot(cx, cy, '+', color='#FF4444', ms=12, mew=1.4)
        label = (f'flux {data["flux"][k]:.0f}\nFWHM {data["fwhm"][k]:.2f}'
                 if panel >= 1.4 else f'{data["fwhm"][k]:.2f}')
        ax.set_title(label, fontsize=fontsize)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in np.ravel(axes)[count:]:
        ax.axis('off')
    fig.suptitle(title or f'frame {frame} -- {count} centroids', fontsize=9,
                 y=1.005)
    fig.tight_layout(pad=0.3)
    return fig


# ------------------------------------------------------------------ step 3

def plot_invariants(points_source, points_target, tolerance=0.02,
                    size=(10, 4)):
    """The two descriptor clouds, and one matched triangle.

    The clouds must overlap. If they do not, the frames share too few stars
    and no amount of tuning will match them.
    """
    from .register import triangle_invariants, match_asterisms
    invariants_source, _ = triangle_invariants(points_source)
    invariants_target, _ = triangle_invariants(points_target)

    fig, axes = plt.subplots(1, 2, figsize=size)
    ax = axes[0]
    ax.scatter(invariants_source[:, 0], invariants_source[:, 1], s=6,
               alpha=0.25, color=ACCENT, label='source')
    ax.scatter(invariants_target[:, 0], invariants_target[:, 1], s=6,
               alpha=0.25, color=WARN, label='target')
    ax.set_xlabel('$d_3/d_2$')
    ax.set_ylabel('$d_2/d_1$')
    ax.set_yscale('log')
    ax.set_title(f'{len(invariants_source)} triangles each, '
                 f'tolerance {tolerance}', fontsize=9)
    ax.legend(fontsize=8, frameon=False)

    correspondences, _ = match_asterisms(points_source, points_target,
                                         tolerance=tolerance)
    ax = axes[1]
    ax.scatter(points_source[:, 1], points_source[:, 0], s=18, color=ACCENT,
               label='source')
    ax.scatter(points_target[:, 1], points_target[:, 0], s=18, marker='s',
               facecolors='none', edgecolors=WARN, label='target')
    for source, target in correspondences:
        ax.plot([points_source[source, 1], points_target[target, 1]],
                [points_source[source, 0], points_target[target, 0]],
                color='#999999', lw=0.6, zorder=0)
    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f'{len(correspondences)} correspondences', fontsize=9)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    return fig


def plot_votes(votes, correspondences=None, size=(5.5, 4.5)):
    """The vote matrix.

    A confident match shows one dominant entry per row. The ratio of the
    winner to the runner-up is the real confidence measure -- about 5x for a
    good match, near 1x when the fields share nothing.
    """
    fig, ax = plt.subplots(figsize=size)
    image = ax.imshow(votes, cmap='magma', origin='upper',
                      interpolation='nearest')
    if correspondences is not None and len(correspondences):
        ax.plot(correspondences[:, 1], correspondences[:, 0], 's',
                mfc='none', mec='#3DDC84', ms=9, mew=1.2,
                label='accepted (mutual best)')
        accepted = votes[correspondences[:, 0], correspondences[:, 1]]
        runner_up = np.sort(votes[correspondences[:, 0]], axis=1)[:, -2]
        margin = np.median(accepted / np.maximum(runner_up, 1))
        ax.set_title(f'median margin over runner-up: {margin:.1f}x',
                     fontsize=9)
        ax.legend(fontsize=8, loc='upper right', framealpha=0.8)
    ax.set_xlabel('target star')
    ax.set_ylabel('source star')
    fig.colorbar(image, ax=ax, label='votes', fraction=0.046)
    fig.tight_layout()
    return fig


def plot_registration_residuals(matched_source, matched_target, scale,
                                rotation, translation, inliers=None,
                                shape=None, size=(11, 3.6)):
    """Residuals against radius and as a vector field.

    Radius dependence means optical distortion, which a similarity cannot
    absorb. A uniform scatter means centroid precision. The vector field
    makes a systematic pattern obvious where a scatter plot would not.
    """
    predicted = (scale * (rotation @ matched_source.T)).T + translation
    offset = predicted - matched_target
    distance = np.linalg.norm(offset, axis=1)
    if inliers is None:
        inliers = np.ones(len(distance), bool)
    centre = (np.array(shape) / 2 if shape is not None
              else matched_source.mean(axis=0))
    radius = np.linalg.norm(matched_source - centre, axis=1)

    fig, axes = plt.subplots(1, 3, figsize=size)
    axes[0].scatter(radius[inliers], distance[inliers], s=22, color=ACCENT)
    if (~inliers).any():
        axes[0].scatter(radius[~inliers], distance[~inliers], s=30,
                        color=WARN, marker='x', label='RANSAC outlier')
        axes[0].legend(fontsize=8, frameon=False)
    axes[0].set_xlabel('distance from field centre (px)')
    axes[0].set_ylabel('residual (px)')
    axes[0].set_title(f'median {np.median(distance[inliers]):.3f} px',
                      fontsize=9)

    axes[1].hist(distance[inliers], bins=12, color=ACCENT)
    axes[1].set_xlabel('residual (px)')
    axes[1].set_ylabel('stars')
    axes[1].set_title('a smooth spread means centroid noise;\n'
                      'a gap means bad correspondences', fontsize=8)

    axes[2].quiver(matched_source[inliers, 1], matched_source[inliers, 0],
                   offset[inliers, 1], offset[inliers, 0],
                   angles='xy', color=ACCENT)
    axes[2].invert_yaxis()
    axes[2].set_aspect('equal')
    axes[2].set_xticks([])
    axes[2].set_yticks([])
    axes[2].set_title('residual vector field', fontsize=9)
    fig.tight_layout()
    return fig


def plot_drift(report, flagged=None, size=(11, 3.4)):
    """Fitted shift and rotation against frame index.

    The Earth turns at a constant rate, so these must be smooth. A frame off
    the curve had a bad fit however small its residual -- the residual only
    says the inliers agree with each other, not that the right stars matched.
    """
    import polars as pl
    ok = report.filter(pl.col('status') == 'ok')
    frame = ok['frame'].to_numpy()
    fig, axes = plt.subplots(1, 3, figsize=size)
    for ax, column, label in zip(axes, ['shift_y', 'shift_x', 'angle'],
                                 ['shift y (px)', 'shift x (px)',
                                  'rotation (deg)']):
        values = ok[column].to_numpy().astype(float)
        ax.plot(frame, values, 'o-', ms=4, lw=1, color=ACCENT)
        if flagged is not None and len(flagged):
            mask = np.isin(frame, flagged)
            ax.plot(frame[mask], values[mask], 'o', ms=10, mfc='none',
                    mec=WARN, mew=1.6)
        ax.set_xlabel('frame')
        ax.set_ylabel(label)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------ step 4

def plot_quality(quality, size=(11, 3.2)):
    """Per-frame metrics, with rejected frames marked.

    Each catches a different failure: star count for cloud, FWHM for focus
    drift, ellipticity for wind or a bumped tripod, flux for transparency.
    """
    import polars as pl
    frame = quality['frame'].to_numpy()
    usable = (quality['usable'].to_numpy().astype(bool)
              if 'usable' in quality.columns
              else np.ones(len(frame), bool))
    columns = [c for c in ('n_stars', 'fwhm', 'ellipticity', 'flux_ratio')
               if c in quality.columns]
    fig, axes = plt.subplots(1, len(columns), figsize=size)
    for ax, column in zip(np.atleast_1d(axes), columns):
        values = quality[column].to_numpy().astype(float)
        ax.plot(frame[usable], values[usable], 'o-', ms=4, lw=1, color=ACCENT)
        if (~usable).any():
            ax.plot(frame[~usable], values[~usable], 'x', ms=9, color=WARN,
                    label='rejected')
            ax.legend(fontsize=8, frameon=False)
        ax.set_xlabel('frame')
        ax.set_ylabel(column)
    fig.tight_layout()
    return fig


def plot_coverage(aligned, box=None, size=(6, 4)):
    """How many frames contribute to each pixel.

    A drifting field leaves the edges covered by far fewer frames than the
    centre, so they are noisier by a factor you did not choose. Cropping to
    the common area gives uniform depth.
    """
    covered = np.isfinite(aligned).sum(axis=0)
    fig, ax = plt.subplots(figsize=size)
    # noinspection PyTypeChecker
    image = ax.imshow(covered[::4, ::4], cmap='viridis', origin='upper',
                      extent=[-0.5, aligned.shape[2] - 0.5,
                              aligned.shape[1] - 0.5, -0.5])
    if box is not None:
        top, bottom, left, right = box
        ax.add_patch(plt.Rectangle((left, top), right - left, bottom - top,
                                   fill=False, ec='#FF4444', lw=1.5,
                                   label='common area'))
        ax.legend(fontsize=8, loc='lower right')
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(image, ax=ax, label='contributing frames', fraction=0.04)
    fig.tight_layout()
    return fig


def plot_stack_comparison(single, combined, detections=None, frame=0,
                          zoom=300, size=(11, 5.5)):
    """A single sub against the stack, whole frame and zoomed."""
    fig, axes = plt.subplots(2, 2, figsize=size)
    show(axes[0, 0], single)
    axes[0, 0].set_title('single sub-exposure', fontsize=9)
    show(axes[0, 1], np.nan_to_num(combined, nan=np.nanmedian(combined)))
    axes[0, 1].set_title('stacked', fontsize=9)

    height, width = single.shape
    y0, x0 = height // 2 - zoom // 2, width // 2 - zoom // 2
    for ax, image, name in [(axes[1, 0], single, 'single'),
                            (axes[1, 1], combined, 'stacked')]:
        cut = image[y0:y0 + zoom, x0:x0 + zoom]
        show(ax, np.nan_to_num(cut, nan=np.nanmedian(cut)), decimate=1)
        ax.set_title(f'{name}, centre {zoom}x{zoom}', fontsize=9)
    fig.tight_layout()
    return fig


def plot_noise_scaling(aligned, size=(6, 4)):
    """Measured noise against the number of frames stacked.

    Uses the half-split estimator, so no ground truth is needed: stack the
    even and odd frames separately, and half their difference is a pure
    noise realisation because the signal cancels exactly. The measured
    points should follow 1/sqrt(N).
    """
    from .stack import stack_frames, half_split_noise
    counts, measured = [], []
    for n in range(2, len(aligned) + 1, max(1, len(aligned) // 8)):
        if n < 2:
            continue
        counts.append(n)
        measured.append(half_split_noise(aligned[:n], method='mean'))
    counts = np.array(counts)
    measured = np.array(measured)

    fig, ax = plt.subplots(figsize=size)
    ax.plot(counts, measured, 'o-', color=ACCENT, label='half-split estimate')
    ax.plot(counts, measured[0] * np.sqrt(counts[0] / counts), '--',
            color=WARN, label='$1/\\sqrt{N}$')
    ax.set_xlabel('frames stacked')
    ax.set_ylabel('noise (ADU)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    return fig
