"""astrostack -- stacking short untracked exposures from raw camera files.

    from astrostack import io, background, detect, register, quality, stack

    cube = io.load_cube(files)
    residual, level, sigma = background.residuals(cube)
    detections = detect.detect_cube(residual, level, sigma)
    frames, table = quality.usable_frames(detections)
    reference = quality.reference_frame(table)
    transforms, report = register.register_all(detections, frames, reference)
    aligned, labels = stack.warp_all(cube, transforms, reference)
    combined, depth = stack.stack_frames(aligned)
"""

from . import (background, detect, dng, exif, io, output,
               plotting, quality, register, stack)

__all__ = ['io', 'background', 'detect', 'register', 'quality', 'stack',
           'output', 'exif', 'plotting', 'dng']
__version__ = '0.1'