"""Build the canvas for the next shard from every shard born so far.

Renders the shard world at the new yaw at full shard resolution and paints
unseen pixels flat gray, plus a white-where-known mask for compose_shard.py.
"""
import argparse
import json

import numpy as np
from PIL import Image

from shardlib import load_shards, render_view

FILL = (128, 128, 128)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", required=True)
    ap.add_argument("--yaw", type=float, required=True)
    ap.add_argument("--pitch", type=float, default=0.0)
    ap.add_argument("--focal-mm", type=float, default=24.0)
    ap.add_argument("--invalid", default="",
                    help="JSON list of [x0,y0,x1,y1] boxes in the first shard to mark unseen")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mask-out", required=True)
    args = ap.parse_args()

    shards = load_shards(json.loads(args.shards))
    for x0, y0, x1, y1 in json.loads(args.invalid or "[]"):
        shards[0][1][y0:y1, x0:x1] = False
    h, w = shards[0][0].shape[:2]
    rgb, known = render_view(shards, args.yaw, w, h, args.focal_mm, args.pitch)
    rgb[~known] = FILL
    Image.fromarray(rgb.clip(0, 255).astype(np.uint8)).save(args.out)
    Image.fromarray((known * 255).astype(np.uint8)).save(args.mask_out)
    print(f"known={known.mean()*100:.1f}%")


if __name__ == "__main__":
    main()
