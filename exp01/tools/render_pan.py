"""Render a camera pan through a set of pure-rotation shards.

Each shard is an RGBA image (alpha 0 = unseen) with a yaw. Shards listed later
take priority where their alpha is valid; ownership was already resolved when
each shard was composed, so overlaps agree. Remaining holes are push-pull filled.
"""
import argparse
import json
import math
import os
import subprocess
import tempfile

import numpy as np
from PIL import Image

from shardlib import bilinear, intrinsics, push_pull_fill, yaw_matrix


def ease(t):
    return 0.5 - 0.5 * math.cos(math.pi * t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", required=True, help='JSON list of {"path":..., "yaw":...}')
    ap.add_argument("--keys", required=True, help="JSON list of yaw keyframes, eased between")
    ap.add_argument("--seconds-per-leg", type=float, default=3.0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--focal-mm", type=float, default=24.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    shards = []
    for s in json.loads(args.shards):
        im = np.asarray(Image.open(s["path"]).convert("RGBA")).astype(np.float64)
        shards.append((im[..., :3], im[..., 3] > 127, s["yaw"]))
    sh, sw = shards[0][0].shape[:2]
    Ks = intrinsics(sw, sh, args.focal_mm)

    ow = args.width
    oh = int(round(ow * sh / sw / 2)) * 2
    Ko = intrinsics(ow, oh, args.focal_mm)
    u, v = np.meshgrid(np.arange(ow) + 0.5, np.arange(oh) + 0.5)
    rays = np.stack([u, v, np.ones_like(u)], -1) @ np.linalg.inv(Ko).T

    keys = json.loads(args.keys)
    n_leg = int(args.seconds_per_leg * args.fps)
    yaws = []
    for a, b in zip(keys[:-1], keys[1:]):
        yaws += [a + (b - a) * ease(i / n_leg) for i in range(n_leg)]
    yaws.append(keys[-1])

    tmp = tempfile.mkdtemp()
    holes = 0.0
    for i, yaw in enumerate(yaws):
        world = rays @ yaw_matrix(yaw).T
        frame = np.zeros((oh, ow, 3))
        have = np.zeros((oh, ow), dtype=bool)
        for rgb, valid, syaw in shards:
            local = world @ yaw_matrix(syaw)  # world -> shard camera (R^T applied as row vectors)
            z = local[..., 2]
            p = local @ Ks.T
            xs = p[..., 0] / z - 0.5
            ys = p[..., 1] / z - 0.5
            inside = (z > 0) & (xs >= 0) & (xs <= sw - 1) & (ys >= 0) & (ys <= sh - 1)
            xi = np.clip(np.round(xs).astype(int), 0, sw - 1)
            yi = np.clip(np.round(ys).astype(int), 0, sh - 1)
            ok = inside & valid[yi, xi]
            frame[ok] = bilinear(rgb, xs, ys)[ok]
            have |= ok
        holes = max(holes, 1 - have.mean())
        frame = push_pull_fill(frame, have)
        Image.fromarray(frame.clip(0, 255).astype(np.uint8)).save(f"{tmp}/{i:05d}.png")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(args.fps),
                    "-i", f"{tmp}/%05d.png", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", "16", args.out], check=True)
    print(f"frames={len(yaws)} max_hole={holes*100:.2f}% -> {args.out}")


if __name__ == "__main__":
    main()
