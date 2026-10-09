"""Measure how much a model-filled shard drifted from its canvas in the known region.

Prints PSNR raw and after blur (texture vs colour/layout) and the worst 64 px
tiles after blur. A worst tile far above a few levels means known structure
changed and the shard should be rejected.
"""
import argparse

import numpy as np
from PIL import Image, ImageFilter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("canvas")
    ap.add_argument("known_mask")
    ap.add_argument("filled")
    args = ap.parse_args()

    c = Image.open(args.canvas).convert("RGB")
    r0 = Image.open(args.filled).convert("RGB")
    r = r0.resize(c.size, Image.LANCZOS)
    m = np.asarray(Image.open(args.known_mask)) > 127
    me = np.asarray(Image.fromarray((m * 255).astype(np.uint8)).filter(ImageFilter.MinFilter(31))) > 127
    print(f"filled size {r0.size}, canvas size {c.size}")
    for rad in (0, 4, 12):
        cb = c.filter(ImageFilter.GaussianBlur(rad)) if rad else c
        rb = r.filter(ImageFilter.GaussianBlur(rad)) if rad else r
        d = (np.asarray(cb).astype(float) - np.asarray(rb).astype(float)) ** 2
        print(f"blur {rad:2d}: PSNR {10 * np.log10(255 ** 2 / d[me].mean()):.1f} dB")
    cb = np.asarray(c.filter(ImageFilter.GaussianBlur(8))).astype(float)
    rb = np.asarray(r.filter(ImageFilter.GaussianBlur(8))).astype(float)
    d = np.abs(cb - rb).mean(-1)
    tiles = []
    for y in range(0, d.shape[0] - 64, 32):
        for x in range(0, d.shape[1] - 64, 32):
            t = me[y:y + 64, x:x + 64]
            if t.mean() > 0.9:
                tiles.append((float(d[y:y + 64, x:x + 64][t].mean()), x, y))
    tiles.sort(reverse=True)
    print("worst tiles (levels, x, y):", [(round(a, 1), x, y) for a, x, y in tiles[:5]])


if __name__ == "__main__":
    main()
