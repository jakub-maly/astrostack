"""Asterism matching and similarity fitting."""

from itertools import combinations
from math import comb

import numpy as np
import polars as pl
from scipy.spatial import KDTree

from .detect import select_stars


def triangle_invariants(points, min_side=1.0, max_triangles=6000, seed=0):
    """Descriptors that survive rotation, translation and scale.

    The transform is what we are solving for, so matching cannot use it --
    we need a quantity computed from positions alone that is unchanged by
    it. A triangle has three sides; dividing out scale leaves two numbers,
    which is exactly its shape up to similarity.

    Vertices come back ordered by the length of the side OPPOSITE each one,
    so two matched triangles have their vertices in corresponding order and
    one triangle match yields three point correspondences for free.

    :return: (invariants (T, 2), vertices (T, 3))
    """
    n_points = len(points)
    if n_points < 3:
        return np.zeros((0, 2)), np.zeros((0, 3), dtype=np.intp)

    triple_index = np.fromiter(combinations(range(n_points), 3),
                               dtype=np.dtype((np.intp, 3)),
                               count=comb(n_points, 3))
    if len(triple_index) > max_triangles:
        generator = np.random.default_rng(seed)
        triple_index = triple_index[generator.choice(len(triple_index),
                                                     max_triangles,
                                                     replace=False)]

    vertex_a = points[triple_index[:, 0]]
    vertex_b = points[triple_index[:, 1]]
    vertex_c = points[triple_index[:, 2]]
    # column k is the side opposite vertex k
    side_lengths = np.stack([np.linalg.norm(vertex_b - vertex_c, axis=1),
                             np.linalg.norm(vertex_a - vertex_c, axis=1),
                             np.linalg.norm(vertex_a - vertex_b, axis=1)],
                            axis=1)

    ascending = np.argsort(side_lengths, axis=1)
    # take_along_axis, not fancy indexing: `ascending` is a DIFFERENT
    # permutation per row, and points[:, ascending] would broadcast the
    # whole index array against every row and give a (T, T, 3) tensor
    sorted_sides = np.take_along_axis(side_lengths, ascending, axis=1)
    vertices = np.take_along_axis(triple_index, ascending, axis=1)

    shortest, middle, longest = sorted_sides.T
    # near-collinear triples have shortest -> 0, so middle/shortest explodes
    # and a sub-pixel centroid error moves the descriptor enormously
    well_conditioned = shortest > min_side
    invariants = np.stack([longest[well_conditioned] / middle[well_conditioned],
                           middle[well_conditioned] / shortest[well_conditioned]],
                          axis=1)
    return invariants, vertices[well_conditioned]


def match_asterisms(points_source, points_target, tolerance=0.02,
                    min_votes=5, **invariant_kwargs):
    """Vote for point correspondences between two star lists.

    A true pair is supported by every triangle containing it whose other two
    vertices are also common to both frames; a chance descriptor collision
    votes once. The decisive filter is not the vote count but *mutual best*
    -- i's favourite is j and j's favourite is i -- which is scale-free and
    needs no tuning.

    :return: (correspondences (M, 2), votes (n_source, n_target))
    """
    invariants_source, vertices_source = triangle_invariants(
        points_source, **invariant_kwargs)
    invariants_target, vertices_target = triangle_invariants(
        points_target, **invariant_kwargs)

    n_source, n_target = len(points_source), len(points_target)
    empty = (np.zeros((0, 2), dtype=np.intp),
             np.zeros((n_source, n_target), dtype=np.int32))
    if len(invariants_source) == 0 or len(invariants_target) == 0:
        return empty

    candidates = KDTree(invariants_target).query_ball_point(
        invariants_source, r=tolerance)
    match_counts = np.fromiter((len(entry) for entry in candidates),
                               dtype=np.intp, count=len(candidates))
    if match_counts.sum() == 0:
        return empty

    source_triangle = np.repeat(np.arange(len(invariants_source)), match_counts)
    target_triangle = np.concatenate([e for e in candidates if e])

    voting_source = vertices_source[source_triangle].ravel()
    voting_target = vertices_target[target_triangle].ravel()
    # bincount on a flattened index, not votes[a, b] += 1: fancy-index
    # assignment increments each repeated pair only ONCE, and the repeats
    # are precisely the signal here
    votes = np.bincount(voting_source * n_target + voting_target,
                        minlength=n_source * n_target)
    votes = votes.reshape(n_source, n_target).astype(np.int32)

    best_target = votes.argmax(axis=1)
    best_source = votes.argmax(axis=0)
    source_indices = np.arange(n_source)
    mutual = ((best_source[best_target] == source_indices) &
              (votes[source_indices, best_target] >= min_votes))
    return np.column_stack([source_indices[mutual],
                            best_target[mutual]]), votes


def fit_similarity(points_source, points_target):
    """Least-squares similarity mapping source onto target (Umeyama 1991).

    Four degrees of freedom in 2-D: one angle, one scale, two shifts. Not
    affine or homography -- the sky does not shear, so extra freedom only
    absorbs centroid noise and distorts the field differently each frame.

    :return: (scale, rotation, translation) with
             target ~= scale * rotation @ source + translation
    """
    points_source = np.asarray(points_source, dtype=float)
    points_target = np.asarray(points_target, dtype=float)
    if points_source.ndim != 2 or points_source.shape[1] != 2:
        raise ValueError(
            f'points_source must be (n, 2), got {points_source.shape}. '
            'Pass positions, not correspondence indices: '
            'points_source[correspondences[:, 0]]')
    if points_source.shape != points_target.shape:
        raise ValueError(f'shape mismatch {points_source.shape} vs '
                         f'{points_target.shape}: the arrays must be '
                         'row-aligned matched pairs')
    if len(points_source) < 2:
        raise ValueError('need at least 2 correspondences')

    n_points = len(points_source)
    centroid_source = points_source.mean(axis=0)
    centroid_target = points_target.mean(axis=0)
    centred_source = points_source - centroid_source
    centred_target = points_target - centroid_target

    # centring decouples translation; the rotation then follows from the
    # SVD of the cross-covariance (the orthogonal Procrustes problem)
    cross_covariance = centred_target.T @ centred_source / n_points
    left, singular_values, right_transposed = np.linalg.svd(cross_covariance)

    # without this the SVD can return a REFLECTION on noisy or degenerate
    # data, silently mirroring the sky
    orientation = np.eye(2)
    if np.linalg.det(left) * np.linalg.det(right_transposed) < 0:
        orientation[-1, -1] = -1

    rotation = left @ orientation @ right_transposed
    variance_source = (centred_source ** 2).sum() / n_points
    scale = float((singular_values * np.diag(orientation)).sum()
                  / max(variance_source, 1e-12))
    translation = centroid_target - scale * rotation @ centroid_source
    return scale, rotation, translation


def fit_similarity_ransac(points_source, points_target, threshold=2.0,
                          iterations=300, seed=0):
    """Similarity fit that ignores wrong correspondences.

    Least squares has an unbounded influence function: one correspondence
    200 px wrong contributes 40000 to the objective while twenty good stars
    at 0.05 px contribute 0.05 between them. Sampling pairs to hypothesise a
    model and keeping the largest agreeing subset holds 0.02 px accuracy
    with 37% of correspondences wrong.

    :return: (scale, rotation, translation, inliers)
    """
    n_points = len(points_source)
    if n_points < 2:
        raise ValueError('need at least 2 correspondences')
    if n_points == 2:
        scale, rotation, translation = fit_similarity(points_source,
                                                      points_target)
        return scale, rotation, translation, np.ones(n_points, dtype=bool)

    generator = np.random.default_rng(seed)
    best_inliers = np.zeros(n_points, dtype=bool)

    for _ in range(iterations):
        sample = generator.choice(n_points, 2, replace=False)
        # a short baseline gives a wildly uncertain rotation angle
        if np.linalg.norm(points_source[sample[0]]
                          - points_source[sample[1]]) < 10.0:
            continue
        try:
            scale, rotation, translation = fit_similarity(
                points_source[sample], points_target[sample])
        except (np.linalg.LinAlgError, ValueError):
            continue
        predicted = (scale * (rotation @ points_source.T)).T + translation
        inliers = np.linalg.norm(predicted - points_target, axis=1) < threshold
        if inliers.sum() > best_inliers.sum():
            best_inliers = inliers
        if best_inliers.all():
            break

    if best_inliers.sum() < 2:
        raise RuntimeError('RANSAC found no consistent similarity transform')

    # the 2-point model is noisy, so refit on its inliers and re-score once
    # with the better transform
    for _ in range(2):
        scale, rotation, translation = fit_similarity(
            points_source[best_inliers], points_target[best_inliers])
        predicted = (scale * (rotation @ points_source.T)).T + translation
        best_inliers = np.linalg.norm(predicted - points_target,
                                      axis=1) < threshold
    return scale, rotation, translation, best_inliers


def register_all(detections, frames, reference, max_stars=25, threshold=2.0,
                 min_inliers=6, **select_kwargs):
    """Fit a transform for every frame onto the reference.

    :return: (transforms, report) -- transforms maps label -> (scale,
             rotation, translation) for frames that succeeded; report has a
             row per frame including the failures
    """
    reference_points = select_stars(detections, reference,
                                    max_stars=max_stars, **select_kwargs)
    if not np.isfinite(reference_points).all():
        raise ValueError(f'reference frame {reference} has non-finite star '
                         'positions; the detection cube still contains NaN')
    transforms, report = {}, []

    for frame in frames:
        frame = int(frame)
        if frame == int(reference):
            transforms[frame] = (1.0, np.eye(2), np.zeros(2))
            report.append(dict(frame=frame, scale=1.0, angle=0.0,
                               shift_y=0.0, shift_x=0.0,
                               n_matched=len(reference_points),
                               n_inliers=len(reference_points),
                               residual=0.0, status='reference'))
            continue

        # one unusable frame must not end the session, so anything that goes
        # wrong here is recorded and skipped rather than raised
        try:
            frame_points = select_stars(detections, frame,
                                        max_stars=max_stars, **select_kwargs)
            correspondences, _ = match_asterisms(frame_points,
                                                 reference_points)
        except (ValueError, RuntimeError) as problem:
            report.append(dict(frame=frame, scale=None, angle=None,
                               shift_y=None, shift_x=None, n_matched=0,
                               n_inliers=0, residual=None,
                               status=f'failed: {problem}'[:60]))
            continue

        blank = dict(frame=frame, scale=None, angle=None, shift_y=None,
                     shift_x=None, n_matched=len(correspondences),
                     n_inliers=0, residual=None)
        if len(correspondences) < 3:
            report.append({**blank, 'status': 'too few matches'})
            continue

        matched_source = frame_points[correspondences[:, 0]]
        matched_target = reference_points[correspondences[:, 1]]
        try:
            scale, rotation, translation, inliers = fit_similarity_ransac(
                matched_source, matched_target, threshold=threshold)
        except (RuntimeError, ValueError, np.linalg.LinAlgError) as problem:
            report.append({**blank, 'status': f'fit failed: {problem}'[:60]})
            continue
        predicted = ((scale * (rotation @ matched_source[inliers].T)).T
                     + translation)
        distances = np.linalg.norm(predicted - matched_target[inliers], axis=1)

        row = dict(frame=frame, scale=float(scale),
                   angle=float(np.degrees(np.arctan2(rotation[1, 0],
                                                     rotation[0, 0]))),
                   shift_y=float(translation[0]),
                   shift_x=float(translation[1]),
                   n_matched=len(correspondences),
                   n_inliers=int(inliers.sum()),
                   residual=float(np.median(distances)))
        if inliers.sum() < min_inliers:
            report.append({**row, 'status': 'too few inliers'})
            continue
        transforms[frame] = (scale, rotation, translation)
        report.append({**row, 'status': 'ok'})

    return transforms, pl.DataFrame(report)


def flag_drift_outliers(report, order=2, kappa=4.0):
    """Frames whose fitted shift departs from the smooth sky drift.

    The Earth turns at a constant rate, so shift against frame index must be
    a smooth curve. A departure is a bad fit however small its residual --
    the residual only says the inliers agree with each other, not that the
    right stars were matched. Order 2 because an alt-az field accelerates
    across the sky.
    """
    usable = report.filter(pl.col('status') == 'ok')
    index = usable['frame'].to_numpy().astype(float)
    flagged = np.zeros(len(index), dtype=bool)
    if len(index) < order + 3:
        return np.zeros(0, dtype=int)

    # two passes: the first outlier inflates the spread and hides milder ones
    for _ in range(2):
        for column in ('shift_y', 'shift_x', 'angle'):
            values = usable[column].to_numpy().astype(float)
            keep = ~flagged
            if keep.sum() < order + 2:
                continue
            coefficients = np.polyfit(index[keep], values[keep], order)
            residual = values - np.polyval(coefficients, index)
            spread = 1.4826 * np.median(np.abs(residual[keep]
                                               - np.median(residual[keep])))
            flagged |= np.abs(residual) > kappa * max(spread, 1e-6)
    return usable['frame'].to_numpy()[flagged]