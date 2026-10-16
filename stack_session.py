#!/usr/bin/env python3
"""Stack a directory of raw frames into a single calibrated colour image.

    python stack_session.py raw-data/ -o m31
    python stack_session.py raw-data/*.CR2 --mono
    python stack_session.py raw-data/ --channels R G1 G2 B

By default this produces a three-plane RGB FITS: the R and B Bayer
sub-lattices plus the mean of G1 and G2, each registered through the whole
session and then aligned to each other.

Output is FITS, not CR2. CR2 is a proprietary container for a compressed
Bayer mosaic plus camera metadata; no library writes one, and a stacked
float32 image with NaNs is none of those things. FITS is what PixInsight,
Siril, DS9 and astropy read.
"""

import argparse
import glob
import os
import sys
import time

import numpy as np

from astrostack import (background, detect, dng, exif, io, output,
                        quality, register, stack)

# 'G' and 'green' both mean the mean of the two green sub-lattices
CHANNEL_ALIASES = {'G': 'green', 'g': 'green', 'green': 'green',
                   'R': 'R', 'r': 'R', 'B': 'B', 'b': 'B',
                   'G1': 'G1', 'g1': 'G1', 'G2': 'G2', 'g2': 'G2'}

# where each sub-lattice samples the sky, in superpixel units relative to R.
# the 'green' average sits midway between G1 and G2.
PLANE_OFFSET = {'R': (0.0, 0.0), 'G1': (0.0, 0.5), 'G2': (0.5, 0.0),
                'green': (0.25, 0.25), 'B': (0.5, 0.5)}


def collect_files(patterns, extensions=('.cr2', '.CR2', '.nef', '.NEF',
                                        '.arw', '.ARW', '.dng', '.DNG')):
    """Expand directories and globs into a sorted file list."""
    files = []
    for pattern in patterns:
        if os.path.isdir(pattern):
            for extension in extensions:
                files.extend(glob.glob(os.path.join(pattern, f'*{extension}')))
        else:
            files.extend(glob.glob(pattern))
    files = sorted(set(files), key=_numeric_key)
    if not files:
        raise SystemExit(f'no raw files matched {patterns}')
    return files


def _numeric_key(path):
    """Sort frame10 after frame9, not after frame1."""
    name = os.path.splitext(os.path.basename(path))[0]
    digits = ''.join(c if c.isdigit() else ' ' for c in name).split()
    return (name.rstrip('0123456789'), int(digits[-1]) if digits else 0, name)


def analyse_channel(cube, args, saturation, log):
    """Find the bad pixels and detect the stars in every frame.

    The cold-pixel map MUST be built before the cold samples are masked:
    once they are NaN, `cube < cold_cut` is False everywhere and the map
    comes back empty. The cube is masked here, in place, once the map has
    been taken.
    """
    bad_pixels, counts = background.cold_pixel_map(cube,
                                                   threshold=args.cold_cut,
                                                   min_frames=args.min_cold_frames)
    cold_samples = int((counts > 0).sum())
    per_frame = float(counts.sum()) / len(cube)
    log(f'  cold pixels: {cold_samples:,} distinct '
        f'({100 * cold_samples / counts.size:.2f} % of the sensor), '
        f'{per_frame:,.0f} per frame')
    log(f'    recurring in >= {args.min_cold_frames} frames: '
        f'{bad_pixels.sum():,} ({100 * bad_pixels.mean():.2f} %)')
    if counts.max() >= 2:
        tally = np.bincount(counts.ravel(), minlength=min(len(cube), 8) + 1)
        spread = '  '.join(f'{k}:{tally[k]:,}'
                           for k in range(1, min(len(tally), 7)))
        log(f'    frames each affected pixel is cold in -> {spread}')

    # reject bad SAMPLES rather than bad pixels: a pixel cold in three frames
    # is still a real measurement in the other thirteen
    cube[cube < args.cold_cut] = np.nan

    # detection needs a cube with no NaN at all -- one inside a star's window
    # makes its centroid NaN, which then poisons the asterism matching
    detection_cube = background.fill_nonfinite(
        background.fill_bad_pixels(cube, bad_pixels))
    residual, level, sigma = background.residuals(detection_cube,
                                                  box=args.box_size)
    log(f'  sigma per frame: {np.round(sigma.ravel(), 1)}')

    detections = detect.detect_cube(residual, level, sigma, box=args.box,
                                    edge=args.box // 2 + 6,
                                    threshold=args.threshold,
                                    saturation=saturation,
                                    bad_pixels=bad_pixels)
    if detections is None:
        raise SystemExit('no stars detected; lower --threshold or check focus')
    flagged = int(detections['on_badpix'].to_numpy().sum())
    log(f'  {len(detections):,} detections, median FWHM '
        f'{np.nanmedian(detections["fwhm"].to_numpy()):.2f} px, '
        f'{flagged} touching a bad pixel'
        + (' (excluded)' if args.exclude_badpix else ' (kept)'))
    return detections, bad_pixels, dict(residual=residual, level=level,
                                        sigma=sigma)


def choose_reference(detections, n_frames, log):
    """Sharpest usable frame from the middle half of the sequence.

    Sharpest because the reference is the only frame never resampled. From
    the middle because on a drifting untracked field it must also overlap
    everything else -- a reference at one end halves the usable session.
    """
    usable, table = quality.usable_frames(detections,
                                          frame_index=np.arange(n_frames))
    for frame, reason in zip(table['frame'].to_numpy(),
                             table['reason'].to_numpy()):
        if reason is not None and reason == reason:
            log(f'  frame {frame} rejected: {reason}')

    frames = table['frame'].to_numpy()
    widths = table['fwhm'].to_numpy().astype(float)
    eligible = np.isin(frames, usable)
    central = eligible & (frames >= n_frames // 4) & (frames <= 3 * n_frames // 4)
    pool = central if central.any() else eligible
    reference = int(frames[pool][np.nanargmin(widths[pool])])
    log(f'  keeping {len(usable)}/{n_frames} frames, reference = {reference}')
    return usable, reference


def write_diagnostics(directory, cube, detections, intermediates, summary,
                      combined, reference, args, log):
    """Write the full set of diagnostic figures for one channel.

    Every stage of the pipeline has a plot that shows something the numbers
    do not: whether the crosses sit on the stars, whether the drift is
    smooth, whether the noise really fell as 1/sqrt(N).
    """
    from astrostack import plotting

    os.makedirs(directory, exist_ok=True)
    residual = intermediates['residual']
    level = float(np.ravel(intermediates['level'])[reference])
    sigma = float(np.ravel(intermediates['sigma'])[reference])
    written = []

    def save(name, figure):
        path = os.path.join(directory, f'{name}.png')
        figure.savefig(path, dpi=110, bbox_inches='tight')
        written.append(os.path.basename(path))

    try:
        save('01_background', plotting.plot_background(
            cube[reference],
            background.compute_background(cube[reference], box=args.box_size)))
        save('02_detection_stages', plotting.plot_detection_stages(
            residual[reference], level, sigma, threshold=args.threshold))
        save('03_detections', plotting.plot_frame(
            cube, detections, frame=reference, max_show=args.plot_stars))
        save('04_centroids', plotting.plot_frame(
            cube, detections, frame=reference, centroids=True,
            n=args.plot_stars, box=max(31, 3 * args.box // 2)))
        save('05_quality', plotting.plot_quality(
            quality.usable_frames(detections,
                                  frame_index=np.arange(len(cube)))[1]))
        save('06_drift', plotting.plot_drift(
            summary['report'],
            register.flag_drift_outliers(summary['report'])))
        save('07_stack_comparison', plotting.plot_stack_comparison(
            cube[reference], combined))
        if len(summary['aligned']) >= 4:
            save('08_noise_scaling',
                 plotting.plot_noise_scaling(summary['aligned']))
    except Exception as problem:
        log(f'  a diagnostic figure failed: {type(problem).__name__}: {problem}')
    log(f'  wrote {len(written)} figures to {directory}/')


def align_and_stack(cube, detections, usable, reference, args, log,
                    transforms=None, report=None):
    """Register the usable frames onto the reference and combine them.

    Pass `transforms` to reuse a geometry measured elsewhere -- which is what
    the Bayer path does, measuring once on the green average and applying the
    same transform to all four sub-planes so they stay on one grid.
    """
    if transforms is None:
        transforms, report = register.register_all(
            detections, usable, reference=reference, max_stars=args.max_stars,
            exclude_saturated=True, exclude_badpix=args.exclude_badpix)
        for frame in register.flag_drift_outliers(report):
            log(f'  frame {frame} departs from the smooth sky drift, dropped')
            transforms.pop(int(frame), None)
    if len(transforms) < 2:
        raise SystemExit('fewer than 2 frames registered')

    succeeded = report.filter(report['status'] == 'ok')
    residuals = (succeeded['residual'].to_numpy().astype(float)
                 if len(succeeded) else np.array([np.nan]))
    median_residual = (float(np.nanmedian(residuals))
                       if np.isfinite(residuals).any() else float('nan'))
    log(f'  registered {len(transforms)} frames, median residual '
        + (f'{median_residual:.3f} px' if median_residual == median_residual
           else 'n/a'))

    aligned, _ = stack.warp_all(cube, transforms, reference=reference,
                                frame_index=np.arange(len(cube)),
                                order=args.order)
    weights = stack.frame_weights(aligned)
    combined, depth = stack.stack_frames(aligned, method=args.combine,
                                         κ=args.kappa, weights=weights)
    noise = stack.half_split_noise(aligned, method='mean')
    single = background.robust_sigma(aligned[0])
    log(f'  noise {single:.2f} -> {noise:.2f} ADU ({single / noise:.2f}x, '
        f'ideal {np.sqrt(len(aligned)):.2f}x)')
    return combined, depth, dict(n_frames=len(aligned),
                                 residual=median_residual, noise=noise,
                                 single_noise=single, report=report,
                                 aligned=aligned, transforms=transforms)


def _shift_plane(image, name, anchor_name, order):
    """Apply the known Bayer sub-lattice offset, with nothing measured."""
    dy = PLANE_OFFSET[anchor_name][0] - PLANE_OFFSET[name][0]
    dx = PLANE_OFFSET[anchor_name][1] - PLANE_OFFSET[name][1]
    return stack.warp_frame(image, 1.0, np.eye(2), np.array([dy, dx]),
                            order=order)


def align_channels(planes, names, args, saturation, log):
    """Bring the colour planes onto a common grid.

    Two things separate them. The Bayer sub-lattices sample the sky at
    positions offset by half a superpixel -- a fixed, known shift. And
    atmospheric dispersion pulls blue and red apart by up to 1.5 arcsec at
    60 degrees from zenith, which is neither fixed nor predictable from the
    geometry. So the offset is measured from the stars, falling back to the
    known geometry only when too few stars match.
    """
    anchor = names.index('green') if 'green' in names else len(names) // 2
    log(f'\naligning channels onto {names[anchor]}:')

    def star_positions(image):
        filled = np.where(np.isfinite(image), image, np.nanmedian(image))
        residual = filled - background.compute_background(filled,
                                                          box=args.box_size)
        found = detect.detect_frame(residual, float(np.median(residual)),
                                    background.robust_sigma(residual),
                                    box=args.box, edge=args.box // 2 + 6,
                                    threshold=args.threshold,
                                    saturation=saturation,
                                    max_stars=args.max_stars)
        if found is None:
            return None
        return np.column_stack([found['y'], found['x']])

    reference_points = star_positions(planes[anchor])
    if reference_points is None or len(reference_points) < 6:
        log('  too few stars in the anchor plane; using the Bayer geometry')
        return [plane if i == anchor
                else _shift_plane(plane, name, names[anchor], args.order)
                for i, (plane, name) in enumerate(zip(planes, names))]

    out = []
    for index, (image, name) in enumerate(zip(planes, names)):
        if index == anchor:
            out.append(image)
            continue
        points = star_positions(image)
        correspondences = (register.match_asterisms(points,
                                                    reference_points)[0]
                           if points is not None and len(points) >= 6
                           else np.zeros((0, 2), dtype=int))
        if len(correspondences) < 4:
            log(f'  {name}: {len(correspondences)} correspondences, '
                'falling back to the geometric offset')
            out.append(_shift_plane(image, name, names[anchor], args.order))
            continue
        scale, rotation, translation, inliers = register.fit_similarity_ransac(
            points[correspondences[:, 0]],
            reference_points[correspondences[:, 1]])
        log(f'  {name}: shift ({translation[0]:+.2f}, {translation[1]:+.2f}) px, '
            f'{inliers.sum()} inliers')
        out.append(stack.warp_frame(image, scale, rotation, translation,
                                    order=args.order))
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('inputs', nargs='+',
                        help='raw files, globs, or a directory')
    parser.add_argument('-o', '--output', default='stack',
                        help='output basename (default: stack)')
    parser.add_argument('--channels', nargs='+', default=['R', 'G', 'B'],
                        metavar='C',
                        help='planes to stack: R, G (mean of G1 and G2), B, '
                             'G1, G2. Default: R G B')
    parser.add_argument('--mono', action='store_true',
                        help='shorthand for --channels G')
    parser.add_argument('--no-align-channels', dest='align_channels',
                        action='store_false',
                        help='skip cross-channel registration')
    parser.add_argument('--box', type=int, default=21,
                        help='star measurement window, about 2.6 x FWHM')
    parser.add_argument('--box-size', type=int, default=64,
                        help='background mesh tile size')
    parser.add_argument('--threshold', type=float, default=5.0,
                        help='detection threshold in sigma')
    parser.add_argument('--max-stars', type=int, default=25,
                        help='stars used for asterism matching')
    parser.add_argument('--combine', default='sigma_clip',
                        choices=['sigma_clip', 'mean', 'median'])
    parser.add_argument('--kappa', type=float, default=3.0,
                        help='sigma-clipping rejection threshold')
    parser.add_argument('--order', type=int, default=3, choices=[0, 1, 3, 5],
                        help='resampling spline order; 3 is almost always right')
    parser.add_argument('--saturation', type=float, default=0.90,
                        help='fraction of full well treated as saturated')
    parser.add_argument('--cold-cut', type=float, default=-50.0,
                        help='ADU below which a sample is discarded')
    parser.add_argument('--min-cold-frames', type=int, default=3,
                        help='frames a pixel must be cold in to enter the '
                             'bad-pixel map. Must be at most the frame '
                             'count, or the map is empty by construction.')
    parser.add_argument('--no-crop', dest='crop', action='store_false',
                        help='keep the full frame instead of the common area')
    parser.add_argument('--exclude-badpix', action='store_true',
                        help='drop stars whose measurement window touches a '
                             'bad pixel. Off by default: the flag fires if '
                             'ANY of the ~441 pixels in the window is '
                             'flagged, so on a sensor with RTS defects most '
                             'stars trip it, while one bad sample among 441 '
                             'barely moves a flux-weighted centroid.')
    parser.add_argument('--plots', action='store_true',
                        help='write the full set of diagnostic figures')
    parser.add_argument('--plot-dir', default=None,
                        help='where the figures go (default: <output>_plots)')
    parser.add_argument('--plot-stars', type=int, default=24,
                        help='stars drawn in the overlay and centroid grid')
    parser.add_argument('--no-fits', dest='fits', action='store_false',
                        help='skip the FITS output. Keep it if you intend to '
                             'measure anything: it is the only format here '
                             'that holds float32, negative sky values, NaN '
                             'coverage gaps and the processing header.')
    parser.add_argument('--no-coverage', dest='coverage_map',
                        action='store_false',
                        help='skip the companion coverage map')
    parser.add_argument('--bayer-dng', action='store_true',
                        help='also write a full-resolution CFA DNG. The '
                             'transform is measured once on green and '
                             'applied to R, G1, G2 and B separately; the '
                             'four stacked planes are then interleaved back '
                             'into a Bayer mosaic with the black level added '
                             'back, so the result is structurally what the '
                             'camera produced, in native ADU. Implies '
                             '--channels R G1 G2 B.')
    parser.add_argument('--dng', action='store_true',
                        help='also write a linear DNG, which Lightroom and '
                             'Darktable open as a raw file')
    parser.add_argument('--tiff', action='store_true', help='also write TIFF')
    parser.add_argument('--no-preview', dest='preview', action='store_false',
                        help='skip the stretched PNG')
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args(argv)

    if args.mono:
        args.channels = ['G']
    if args.bayer_dng:
        # the mosaic needs the two greens kept apart: writing one averaged
        # green to both sites would fabricate a correlation no sensor shows
        args.channels = ['R', 'G1', 'G2', 'B']
        args.crop = False       # a mosaic must keep its 2x2 phase intact
    try:
        channels = [CHANNEL_ALIASES[c] for c in args.channels]
    except KeyError as bad:
        raise SystemExit(f'unknown channel {bad}; choose from R, G, B, G1, G2')
    if len(set(channels)) != len(channels):
        raise SystemExit(f'duplicate channels in {args.channels}')

    if not any((args.fits, args.bayer_dng, args.dng, args.tiff,
                args.preview, args.plots)):
        raise SystemExit('--no-fits leaves nothing to write; add --bayer-dng, '
                         '--dng, --tiff, --preview or --plots')

    log = (lambda *a: None) if args.quiet else (lambda *a: print(*a, flush=True))
    started = time.time()

    files = collect_files(args.inputs)
    if args.min_cold_frames > len(files):
        log(f'note: --min-cold-frames {args.min_cold_frames} exceeds the '
            f'{len(files)} frames available; lowering it to {len(files)}')
        args.min_cold_frames = len(files)
    log(f'{len(files)} raw files, {os.path.basename(files[0])} .. '
        f'{os.path.basename(files[-1])}')
    metadata = io.load_metadata(files[0])
    saturation = io.saturation_level(metadata, fraction=args.saturation)
    camera = exif.read_exif(files[0])
    if camera:
        log(f'{camera.get("model", "?")}, {camera.get("lens_model", "?")} at '
            f'{camera.get("focal_length", "?")} mm f/{camera.get("f_number", "?")}, '
            f'ISO {camera.get("iso", "?")}, {camera.get("exposure_time", "?")} s')
    log(f'white level {metadata["white_level"]}, saturation cut '
        f'{saturation:.0f} ADU, channels {" ".join(channels)}')

    # the anchor channel is analysed first and fixes the reference frame and
    # the frame selection for every channel, so all planes share one grid
    planes, depths, summaries = {}, {}, {}
    reference = usable = None
    shared_transforms = shared_report = None

    if args.bayer_dng:
        # measure the geometry once, on the mean of the two greens: it has the
        # most photons and the finest sampling, so it gives the best stars.
        # Every sub-plane then gets the SAME transform, which is what keeps
        # them on one grid and lets them interleave back into a valid mosaic.
        log('\nmeasuring the transform on the green average:')
        green = io.load_cube(files, channel='green')
        green_detections, _, _ = analyse_channel(green, args, saturation, log)
        usable, reference = choose_reference(green_detections, len(green), log)
        shared_transforms, shared_report = register.register_all(
            green_detections, usable, reference=reference,
            max_stars=args.max_stars, exclude_saturated=True,
            exclude_badpix=args.exclude_badpix)
        for frame in register.flag_drift_outliers(shared_report):
            log(f'  frame {frame} departs from the smooth sky drift, dropped')
            shared_transforms.pop(int(frame), None)
        residuals = shared_report.filter(
            shared_report['status'] == 'ok')['residual'].to_numpy()
        residuals = np.asarray(residuals, dtype=float)
        log(f'  {len(shared_transforms)} frames registered, median residual '
            + (f'{np.nanmedian(residuals):.3f} px'
               if np.isfinite(residuals).any() else 'n/a'))
        del green

    anchor = 'green' if 'green' in channels else channels[0]
    for channel in [anchor] + [c for c in channels if c != anchor]:
        log(f'\nchannel {channel}:')
        cube = io.load_cube(files, channel=channel)
        if shared_transforms is not None:
            detections, intermediates = green_detections, None
            cube[cube < args.cold_cut] = np.nan
            log('  applying the green transform, no re-registration')
        else:
            detections, _, intermediates = analyse_channel(cube, args,
                                                           saturation, log)
            if reference is None:
                usable, reference = choose_reference(detections, len(cube), log)
            else:
                log(f'  reusing reference {reference} from {anchor}')
        planes[channel], depths[channel], summaries[channel] = align_and_stack(
            cube, detections, usable, reference, args, log,
            transforms=shared_transforms, report=shared_report)
        if args.plots and intermediates is not None:
            plot_dir = args.plot_dir or f'{args.output}_plots'
            write_diagnostics(os.path.join(plot_dir, channel), cube,
                              detections, intermediates, summaries[channel],
                              planes[channel], reference, args, log)
        # the aligned cube is only needed for the figures, and it is large
        summaries[channel].pop('aligned', None)
        del cube

    stacked = [planes[c] for c in channels]     # back to the requested order
    if args.bayer_dng:
        # a CFA grid locks the channels to the mosaic, so a per-channel shift
        # cannot be expressed. Leaving the dispersion in is also the faithful
        # choice: a real exposure has it, and readers have their own lateral
        # chromatic aberration correction.
        log('\nskipping cross-channel alignment: the mosaic keeps the '
            'channels on one grid')
    elif len(stacked) > 1 and args.align_channels:
        stacked = align_channels(stacked, channels, args, saturation, log)

    image = np.stack(stacked) if len(stacked) > 1 else stacked[0]
    coverage = np.minimum.reduce([depths[c] for c in channels])

    if args.crop:
        probe = np.stack([np.where(np.isfinite(p), 0.0, np.nan)
                          for p in stacked])
        _, box = stack.crop_to_common(probe, min_frames=len(probe))
        top, bottom, left, right = box
        image = image[..., top:bottom, left:right]
        coverage = coverage[top:bottom, left:right]
        log(f'\ncropped to {image.shape[-2]}x{image.shape[-1]}')

    first = summaries[channels[0]]
    header = {
        'ORIGIN': ('astrostack', 'pipeline that produced this file'),
        'NFRAMES': (first['n_frames'], 'frames combined'),
        'NINPUT': (len(files), 'frames offered'),
        'REFFRAME': (reference, 'registration reference'),
        'COMBINE': (args.combine, 'combination method'),
        'KAPPA': (args.kappa, 'sigma-clip rejection threshold'),
        'RESAMPLE': (args.order, 'spline order'),
        'DETBOX': (args.box, 'star measurement window, px'),
        'DETSIG': (args.threshold, 'detection threshold, sigma'),
        'REGRESID': (round(first['residual'], 4)
                     if first['residual'] == first['residual'] else 0.0,
                     'median registration residual, px'),
        'NOISE1': (round(first['single_noise'], 3), 'single frame noise, ADU'),
        'NOISEN': (round(first['noise'], 3), 'stack noise, half-split, ADU'),
        'GAINSNR': (round(first['single_noise'] / first['noise'], 3),
                    'noise improvement'),
        'CHANNELS': (' '.join(channels), 'planes, in array order'),
        'CHALIGN': (bool(len(channels) > 1 and args.align_channels),
                    'planes cross-registered'),
        'BUNIT': ('ADU', 'linear, black level subtracted'),
        'DEMOSAIC': ('superpixel', 'one 2x2 Bayer tile per output pixel'),
        'WHITELEV': (metadata['white_level'], 'sensor saturation, raw ADU'),
        'BLACKLEV': (round(float(np.median(metadata['black_level'])), 1),
                     'pedestal subtracted, ADU'),
        'COMMENT': 'NaN marks pixels no frame covered',
    }

    # camera settings from the reference frame's EXIF, so the result carries
    # the conditions it was taken under
    reference_file = files[min(reference, len(files) - 1)]
    camera_header = exif.exif_to_fits_header(exif.read_exif(reference_file))
    header.update(camera_header)
    if 'EXPTIME' in camera_header:
        header['TOTALEXP'] = (round(camera_header['EXPTIME'][0]
                                    * first['n_frames'], 1),
                              'total integration, seconds')
    pitch = exif.sensor_pitch(camera)
    if pitch and camera.get('focal_length'):
        header['XPIXSZ'] = (round(pitch * 2, 3),
                            'superpixel size, microns')
        header['SECPIX'] = (round(exif.pixel_scale(camera['focal_length'],
                                                   pitch), 3),
                            'plate scale, arcsec per pixel')

    if args.fits:
        fits_path = f'{args.output}.fits'
        output.write_fits(fits_path, image, header)
        log(f'\nwrote {fits_path}  {image.shape} float32 '
            f'({os.path.getsize(fits_path) / 1e6:.0f} MB)')
        if args.coverage_map:
            output.write_fits(f'{args.output}_coverage.fits',
                              coverage.astype(np.float32),
                              {'COMMENT': 'frames contributing to each pixel'})

    if args.bayer_dng:
        planes_named = {name: planes[name] for name in ('R', 'G1', 'G2', 'B')}
        black_by_name = {}
        pattern = np.asarray(metadata['pattern'])
        green_count = 0
        for index in range(4):
            letter = metadata['colour_description'][index] \
                if index < len(metadata['colour_description']) else 'RGBG'[index]
            if letter == 'G':
                green_count += 1
                black_by_name[f'G{green_count}'] = metadata['black_level'][index]
            else:
                black_by_name[letter] = metadata['black_level'][index]
        mosaic, cfa = dng.interleave_bayer(planes_named, pattern,
                                           black_by_name,
                                           white_level=metadata['white_level'])
        try:
            matrix, balance = dng.colour_profile(files[0])
        except Exception:
            matrix, balance = None, None
        bayer_path = dng.write_bayer_dng(
            f'{args.output}_bayer.dng', mosaic, cfa,
            black_level=float(np.median(metadata['black_level'])),
            white_level=metadata['white_level'], colour_matrix=matrix,
            white_balance=balance,
            camera_model=camera.get('model', 'astrostack'),
            datetime_original=camera.get('datetime_original'),
            original_name=os.path.basename(files[0]))
        log(f'wrote {bayer_path}  {mosaic.shape[0]}x{mosaic.shape[1]} CFA, '
            f'native ADU, black {np.median(metadata["black_level"]):.0f} '
            f'white {metadata["white_level"]}')
    if args.dng:
        if image.ndim != 3 or image.shape[0] != 3:
            log('skipping DNG: it needs exactly three planes (R G B)')
        else:
            try:
                matrix, balance = dng.colour_profile(files[0])
            except Exception as problem:
                matrix, balance = None, None
                log(f'  no colour profile from the raw ({problem}); '
                    'the DNG will render with the reader\'s default colour')
            dng_path, dng_scale = dng.write_dng(
                f'{args.output}.dng', image, colour_matrix=matrix,
                white_balance=balance,
                camera_model=camera.get('model', 'astrostack'),
                datetime_original=camera.get('datetime_original'),
                original_name=os.path.basename(files[0]))
            log(f'wrote {dng_path}  (16-bit linear; ADU = value / '
                f'{dng_scale:.5f})')
    if args.tiff:
        log(f'wrote {output.write_tiff(f"{args.output}.tif", image)}')
    if args.preview:
        log(f'wrote {output.write_preview(f"{args.output}.png", image)}')
    log(f'\ndone in {time.time() - started:.0f} s')
    return 0


if __name__ == '__main__':
    sys.exit(main())