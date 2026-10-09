"""Pure-rotation canvas builder for the observation-shard experiment.

Given a source shard and a yaw angle, re-projects the shard into a camera that
stands at the same point but is rotated. Pixels the source never saw are left
as a flat fill colour so the image model can paint them in.

Pure rotation needs no depth: the mapping is the homography K R K^-1.
"""
import argparse
import json
import math

import numpy as np
from PIL import Image

FILL = (128, 128, 128)


def intrinsics(w, h, focal_mm, sensor_w_mm=36.0):
    f = focal_mm / sensor_w_mm * w
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], dtype=np.float64)


def yaw_matrix(deg):
    # Positive yaw turns the new camera to the right of the source camera.
    t = math.radians(deg)
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def bilinear(img, x, y):
    h, w, _ = img.shape
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    x1, y1 = x0 + 1, y0 + 1
    wx, wy = (x - x0)[..., None], (y - y0)[..., None]
    x0c, x1c = np.clip(x0, 0, w - 1), np.clip(x1, 0, w - 1)
    y0c, y1c = np.clip(y0, 0, h - 1), np.clip(y1, 0, h - 1)
    a = img[y0c, x0c] * (1 - wx) + img[y0c, x1c] * wx
    b = img[y1c, x0c] * (1 - wx) + img[y1c, x1c] * wx
    return a * (1 - wy) + b * wy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--yaw", type=float, required=True)
    ap.add_argument("--focal-mm", type=float, default=24.0)
    ap.add_argument("--invalid", type=str, default="",
                    help="JSON list of [x0,y0,x1,y1] boxes in the source to treat as unseen")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mask-out", required=True)
    args = ap.parse_args()

    src = np.asarray(Image.open(args.src).convert("RGB")).astype(np.float64)
    h, w, _ = src.shape
    valid_src = np.ones((h, w), dtype=bool)
    for x0, y0, x1, y1 in json.loads(args.invalid or "[]"):
        valid_src[y0:y1, x0:x1] = False

    K = intrinsics(w, h, args.focal_mm)
    H = K @ yaw_matrix(args.yaw) @ np.linalg.inv(K)

    u, v = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    pts = np.stack([u, v, np.ones_like(u)], axis=-1) @ H.T
    z = pts[..., 2]
    xs = pts[..., 0] / z - 0.5
    ys = pts[..., 1] / z - 0.5

    inside = (z > 0) & (xs >= 0) & (xs <= w - 1) & (ys >= 0) & (ys <= h - 1)
    xi = np.clip(np.round(xs).astype(int), 0, w - 1)
    yi = np.clip(np.round(ys).astype(int), 0, h - 1)
    known = inside & valid_src[yi, xi]

    out = np.empty_like(src)
    out[:] = FILL
    out[known] = bilinear(src, xs, ys)[known]

    Image.fromarray(out.clip(0, 255).astype(np.uint8)).save(args.out)
    Image.fromarray((known * 255).astype(np.uint8)).save(args.mask_out)
    print(f"focal_px={K[0,0]:.1f} hfov={math.degrees(2*math.atan(w/2/K[0,0])):.1f}deg "
          f"known={known.mean()*100:.1f}%")


if __name__ == "__main__":
    main()
