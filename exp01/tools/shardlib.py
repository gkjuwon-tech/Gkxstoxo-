"""Shared helpers: camera model, yaw rotation, bilinear sampling, hole fill."""
import math

import numpy as np


def intrinsics(w, h, focal_mm, sensor_w_mm=36.0):
    f = focal_mm / sensor_w_mm * w
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], dtype=np.float64)


def yaw_matrix(deg):
    """Rotation taking a ray in a camera yawed by `deg` (right positive) to the world frame."""
    t = math.radians(deg)
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def bilinear(img, x, y):
    h, w = img.shape[:2]
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    wx, wy = x - x0, y - y0
    if img.ndim == 3:
        wx, wy = wx[..., None], wy[..., None]
    x0c, x1c = np.clip(x0, 0, w - 1), np.clip(x0 + 1, 0, w - 1)
    y0c, y1c = np.clip(y0, 0, h - 1), np.clip(y0 + 1, 0, h - 1)
    a = img[y0c, x0c] * (1 - wx) + img[y0c, x1c] * wx
    b = img[y1c, x0c] * (1 - wx) + img[y1c, x1c] * wx
    return a * (1 - wy) + b * wy


def push_pull_fill(img, valid, levels=8):
    """Fill invalid pixels by smooth interpolation from valid neighbours."""
    img = img.astype(np.float64)
    if valid.all():
        return img
    w = valid.astype(np.float64)
    pyr = [(img * w[..., None], w)]
    for _ in range(levels):
        a, b = pyr[-1]
        h2, w2 = (a.shape[0] + 1) // 2, (a.shape[1] + 1) // 2
        a = np.pad(a, ((0, h2 * 2 - a.shape[0]), (0, w2 * 2 - a.shape[1]), (0, 0)))
        b = np.pad(b, ((0, h2 * 2 - b.shape[0]), (0, w2 * 2 - b.shape[1])))
        a = a.reshape(h2, 2, w2, 2, -1).sum((1, 3))
        b = b.reshape(h2, 2, w2, 2).sum((1, 3))
        pyr.append((a, b))
    a, b = pyr[-1]
    cur = a / np.maximum(b, 1e-9)[..., None]
    for a, b in reversed(pyr[:-1]):
        up = np.repeat(np.repeat(cur, 2, 0), 2, 1)[: a.shape[0], : a.shape[1]]
        wb = np.clip(b, 0, 1)[..., None]
        cur = np.where(b[..., None] > 0, a / np.maximum(b, 1e-9)[..., None], up) * wb + up * (1 - wb)
    return np.where(valid[..., None], img, cur)


def load_shards(specs):
    """specs: list of {"path": RGBA png, "yaw": degrees}. Later entries win on overlap."""
    from PIL import Image

    out = []
    for s in specs:
        im = np.asarray(Image.open(s["path"]).convert("RGBA")).astype(np.float64)
        out.append((im[..., :3], im[..., 3] > 127, float(s["yaw"])))
    return out


def render_view(shards, yaw, ow, oh, focal_mm):
    """Render the shard world from a camera at `yaw`. Returns (rgb, known_mask)."""
    sh, sw = shards[0][0].shape[:2]
    Ks = intrinsics(sw, sh, focal_mm)
    Ko = intrinsics(ow, oh, focal_mm)
    u, v = np.meshgrid(np.arange(ow) + 0.5, np.arange(oh) + 0.5)
    rays = np.stack([u, v, np.ones_like(u)], -1) @ np.linalg.inv(Ko).T
    world = rays @ yaw_matrix(yaw).T
    frame = np.zeros((oh, ow, 3))
    have = np.zeros((oh, ow), dtype=bool)
    for rgb, valid, syaw in shards:
        local = world @ yaw_matrix(syaw)
        z = local[..., 2]
        p = local @ Ks.T
        with np.errstate(divide="ignore", invalid="ignore"):
            xs = p[..., 0] / z - 0.5
            ys = p[..., 1] / z - 0.5
        inside = (z > 0) & (xs >= 0) & (xs <= sw - 1) & (ys >= 0) & (ys <= sh - 1)
        xs = np.where(inside, xs, 0)
        ys = np.where(inside, ys, 0)
        xi = np.clip(np.round(xs).astype(int), 0, sw - 1)
        yi = np.clip(np.round(ys).astype(int), 0, sh - 1)
        ok = inside & valid[yi, xi]
        frame[ok] = bilinear(rgb, xs, ys)[ok]
        have |= ok
    return frame, have
