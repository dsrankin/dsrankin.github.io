#!/usr/bin/env python3
"""Build the header logo (images/logo/nca.webp + nca.png) from the NCA growth GIF.

Usage: python scripts/make_logo.py path/to/full_neuralcellautomaton_logo.gif

The source GIF is 104px pixel art upscaled 4x on an opaque white background.
This downsamples it back to 104px (the page upscales with image-rendering:
pixelated), makes the white background touching the border transparent (the
white glow in the middle stays), and encodes a lossless animated WebP with
img2webp (libwebp). Alpha stays fully on/off so frames can be stored as small
deltas. nca.png is the final frame, used for reduced motion and as a fallback.
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageSequence
from scipy import ndimage

SCALE = 4
LIGHT = 170  # pixels with every channel >= this count as background if they touch the border
FRAME_MS = 40
OUT = Path(__file__).resolve().parent.parent / 'images' / 'logo'


def to_rgba(frame):
    rgb = np.asarray(frame.convert('RGB'))[::SCALE, ::SCALE]
    labels, _ = ndimage.label(rgb.min(axis=2) >= LIGHT)
    edge = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    background = np.isin(labels, edge[edge > 0])
    rgba = np.dstack([rgb, np.full(rgb.shape[:2], 255, np.uint8)])
    rgba[background] = 0
    return Image.fromarray(rgba, 'RGBA')


def main(src):
    frames = [to_rgba(f) for f in ImageSequence.Iterator(Image.open(src))]
    OUT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        paths = []
        for i, frame in enumerate(frames):
            paths.append(f'{tmp}/{i:04d}.png')
            frame.save(paths[-1])
        subprocess.run(['img2webp', '-loop', '0', '-min_size', '-lossless', '-d', str(FRAME_MS),
                        *paths, '-o', str(OUT / 'nca.webp')], check=True, capture_output=True)
    frames[-1].save(OUT / 'nca.png', optimize=True)


if __name__ == '__main__':
    main(sys.argv[1])
