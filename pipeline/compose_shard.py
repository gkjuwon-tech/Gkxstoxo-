"""Apply the first-come ownership rule to a model-filled shard.

Inside the region the canvas already knew, the older shard's pixels win; the
model's pixels are only kept where the canvas was gray. A feather band just
inside the known region hides the seam. Boxes passed with --invalid (such as a
generator watermark) are marked unseen in the output alpha.
"""
import argparse
import json

import numpy as np
from PIL import Image, ImageFilter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("canvas")
    ap.add_argument("known_mask")
    ap.add_argument("filled")
    ap.add_argument("--feather", type=int, default=24)
    ap.add_argument("--invalid", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    canvas = Image.open(args.canvas).convert("RGB")
    filled = Image.open(args.filled).convert("RGB").resize(canvas.size, Image.LANCZOS)
    known = Image.open(args.known_mask).convert("L")

    f = args.feather
    inner = known.filter(ImageFilter.MinFilter(2 * f + 1)).filter(ImageFilter.GaussianBlur(f / 2))
    alpha = np.asarray(inner).astype(np.float64)[..., None] / 255.0

    C = np.asarray(canvas).astype(np.float64)
    F = np.asarray(filled).astype(np.float64)
    out = C * alpha + F * (1 - alpha)

    valid = np.full(out.shape[:2], 255, dtype=np.uint8)
    for x0, y0, x1, y1 in json.loads(args.invalid or "[]"):
        valid[y0:y1, x0:x1] = 0

    rgba = np.dstack([out.clip(0, 255).astype(np.uint8), valid])
    Image.fromarray(rgba, "RGBA").save(args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
