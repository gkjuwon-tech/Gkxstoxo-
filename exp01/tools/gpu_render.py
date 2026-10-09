"""GPU renderer for a pure-rotation shard world.

All shards live on the GPU as one texture stack. Each frame builds one sampling
grid per shard from the camera rotation and samples every shard in a single
batched call; later shards win where they are valid, as in render_pan.py.
Remaining holes are filled with a small GPU push-pull pyramid.

Runs on CUDA when available, otherwise on CPU.

Modes:
  render  pan through shards from disk and write an mp4
  bench   build a synthetic 360-degree world in memory and time the renderer
"""
import argparse
import json
import math
import os
import shutil
import subprocess
import time

import numpy as np
import torch
import torch.nn.functional as F


def intrinsics(w, h, focal_mm, sensor_w_mm=36.0):
    f = focal_mm / sensor_w_mm * w
    return torch.tensor([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], dtype=torch.float32)


def yaw_matrix(deg):
    """Rotation taking a ray in a camera yawed by `deg` (right positive) to the world frame."""
    t = math.radians(deg)
    c, s = math.cos(t), math.sin(t)
    return torch.tensor([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=torch.float32)


def push_pull_fill(rgb, valid, levels=8):
    """rgb: [3,H,W], valid: [H,W] bool. Fills invalid pixels from valid neighbours."""
    if bool(valid.all()):
        return rgb
    w = valid.float()[None, None]
    a = (rgb[None] * w)
    pyr = [(a, w)]
    for _ in range(levels):
        a, w = pyr[-1]
        if min(a.shape[-2:]) < 2:
            break
        pyr.append((F.avg_pool2d(a, 2, ceil_mode=True) * 4, F.avg_pool2d(w, 2, ceil_mode=True) * 4))
    a, w = pyr[-1]
    cur = a / w.clamp_min(1e-6)
    for a, w in reversed(pyr[:-1]):
        up = F.interpolate(cur, size=a.shape[-2:], mode="bilinear", align_corners=False)
        here = a / w.clamp_min(1e-6)
        k = w.clamp(0, 1)
        cur = here * k + up * (1 - k)
    return torch.where(valid[None], rgb, cur[0])


class ShardWorld:
    def __init__(self, textures, yaws, focal_mm, device):
        """textures: [N,4,H,W] float in 0..1 (alpha 1 = seen). Later shards win."""
        self.device = device
        self.tex = textures.to(device)
        self.n, _, self.sh, self.sw = self.tex.shape
        self.Ks = intrinsics(self.sw, self.sh, focal_mm).to(device)
        # world -> shard camera for every shard: R(syaw)^T
        self.Rs = torch.stack([yaw_matrix(y).T for y in yaws]).to(device)
        self.focal_mm = focal_mm
        self._rays = {}

    def rays(self, ow, oh):
        key = (ow, oh)
        if key not in self._rays:
            Ko = intrinsics(ow, oh, self.focal_mm).to(self.device)
            v, u = torch.meshgrid(torch.arange(oh, device=self.device) + 0.5,
                                  torch.arange(ow, device=self.device) + 0.5, indexing="ij")
            pix = torch.stack([u, v, torch.ones_like(u)], -1)
            self._rays[key] = pix @ torch.linalg.inv(Ko).T
        return self._rays[key]

    @torch.no_grad()
    def render(self, yaw, ow, oh, fill=True):
        """Returns ([3,oh,ow] float 0..1, [oh,ow] bool known)."""
        world = self.rays(ow, oh) @ yaw_matrix(yaw).to(self.device).T          # [oh,ow,3]
        local = torch.einsum("hwc,ndc->nhwd", world, self.Rs)                  # [N,oh,ow,3]
        p = local @ self.Ks.T
        z = p[..., 2]
        front = z > 1e-6
        zs = torch.where(front, z, torch.ones_like(z))
        gx = (p[..., 0] / zs) / self.sw * 2 - 1
        gy = (p[..., 1] / zs) / self.sh * 2 - 1
        grid = torch.stack([gx, gy], -1)
        grid = torch.where(front[..., None], grid, torch.full_like(grid, 3.0))
        samp = F.grid_sample(self.tex, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        ok = samp[:, 3] > 0.5                                                   # [N,oh,ow]
        rgb = samp[:, :3] / samp[:, 3:4].clamp_min(1e-6)                       # un-premultiply edge blend
        # later shards win: take the last valid index per pixel
        idx = torch.arange(self.n, device=self.device)[:, None, None].expand_as(ok)
        last = torch.where(ok, idx, torch.full_like(idx, -1)).amax(0)          # [oh,ow]
        known = last >= 0
        gather = last.clamp_min(0)[None, None].expand(1, 3, oh, ow)
        out = torch.gather(rgb.permute(1, 0, 2, 3), 1, gather.permute(1, 0, 2, 3)).squeeze(1)
        out = torch.where(known[None], out, torch.zeros_like(out))
        if fill:
            out = push_pull_fill(out, known)
        return out.clamp(0, 1), known


def ease(t):
    return 0.5 - 0.5 * math.cos(math.pi * t)


def yaw_path(keys, frames_per_leg):
    yaws = []
    for a, b in zip(keys[:-1], keys[1:]):
        yaws += [a + (b - a) * ease(i / frames_per_leg) for i in range(frames_per_leg)]
    yaws.append(keys[-1])
    return yaws


def open_writer(path, ow, oh, fps):
    exe = shutil.which("ffmpeg")
    if exe is None:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
    return subprocess.Popen([exe, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                             "-s", f"{ow}x{oh}", "-r", str(fps), "-i", "-", "-c:v", "libx264",
                             "-pix_fmt", "yuv420p", "-crf", "16", path], stdin=subprocess.PIPE)


def write_video(world, yaws, ow, oh, fps, path):
    proc = open_writer(path, ow, oh, fps)
    t0 = time.time()
    for y in yaws:
        rgb, _ = world.render(y, ow, oh)
        proc.stdin.write((rgb * 255 + 0.5).byte().permute(1, 2, 0).contiguous().cpu().numpy().tobytes())
    proc.stdin.close()
    proc.wait()
    return time.time() - t0


def load_textures(specs):
    from PIL import Image

    texs, yaws = [], []
    for s in specs:
        im = np.asarray(Image.open(s["path"]).convert("RGBA")).astype(np.float32) / 255.0
        a = (im[..., 3:] > 0.5).astype(np.float32)
        im = np.concatenate([im[..., :3] * a, a], -1)  # premultiplied
        texs.append(torch.from_numpy(im).permute(2, 0, 1))
        yaws.append(float(s["yaw"]))
    return torch.stack(texs), yaws


def synthetic_world(n, sw, sh, focal_mm, device):
    """Shards cut from a procedural 360-degree panorama, every 360/n degrees."""
    Ks = intrinsics(sw, sh, focal_mm).to(device)
    v, u = torch.meshgrid(torch.arange(sh, device=device) + 0.5,
                          torch.arange(sw, device=device) + 0.5, indexing="ij")
    rays = torch.stack([u, v, torch.ones_like(u)], -1) @ torch.linalg.inv(Ks).T
    texs, yaws = [], []
    for i in range(n):
        yaw = 360.0 * i / n
        d = rays @ yaw_matrix(yaw).to(device).T
        d = d / d.norm(dim=-1, keepdim=True)
        lon = torch.atan2(d[..., 0], d[..., 2])
        lat = torch.asin(d[..., 1].clamp(-1, 1))
        grid = ((torch.cos(lon * 72) > 0.995) | (torch.cos(lat * 72) > 0.995)).float()
        r = 0.5 + 0.5 * torch.sin(lon)
        g = 0.5 + 0.5 * torch.sin(lon * 3 + 2) * torch.cos(lat * 4)
        b = 0.5 + 0.5 * torch.cos(lon * 5 + lat * 7)
        rgb = torch.stack([r, g, b]) * (1 - grid) + grid
        a = torch.ones(1, sh, sw, device=device)
        a[:, int(sh * 0.9):int(sh * 0.95), int(sw * 0.94):int(sw * 0.98)] = 0  # fake watermark hole
        texs.append(torch.cat([rgb * a, a]).cpu())
        yaws.append(yaw)
    return torch.stack(texs), yaws


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["render", "bench"])
    ap.add_argument("--shards", help='render: JSON list of {"path":..., "yaw":...}')
    ap.add_argument("--keys", default="[0, 360]", help="JSON yaw keyframes, eased between")
    ap.add_argument("--seconds-per-leg", type=float, default=6.0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--focal-mm", type=float, default=24.0)
    ap.add_argument("--half", action="store_true", help="store textures in float16")
    ap.add_argument("--out", default="pan.mp4")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    info = {"device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu"}

    if args.mode == "render":
        texs, yaws = load_textures(json.loads(args.shards))
    else:
        texs, yaws = synthetic_world(12, 2752, 1536, args.focal_mm, device)
    if args.half:
        texs = texs.half()
    world = ShardWorld(texs, yaws, args.focal_mm, device)
    if args.half:
        world.tex = world.tex.half()
    ow, oh = args.width, args.height
    path = yaw_path(json.loads(args.keys), int(args.seconds_per_leg * args.fps))

    if args.mode == "bench":
        def sync():
            if device == "cuda":
                torch.cuda.synchronize()
        for y in path[:5]:
            world.render(y, ow, oh)
        sync()
        for fill in (True, False):
            t0 = time.time()
            for y in path:
                world.render(y, ow, oh, fill=fill)
            sync()
            dt = time.time() - t0
            info[f"render_fps_fill_{fill}"] = round(len(path) / dt, 1)
        t0 = time.time()
        for y in path:
            rgb, _ = world.render(y, ow, oh)
            (rgb * 255 + 0.5).byte().permute(1, 2, 0).contiguous().cpu()
        sync()
        info["render_plus_download_fps"] = round(len(path) / (time.time() - t0), 1)
        a, _ = world.render(path[0], ow, oh)
        b, _ = world.render(path[-1], ow, oh)
        info["loop_closure_max_abs_diff_8bit"] = int(((a - b).abs().max() * 255).round())
        if device == "cuda":
            info["gpu_mem_peak_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
        info["shards"] = world.n
        info["frames"] = len(path)
        info["resolution"] = f"{ow}x{oh}"

    secs = write_video(world, path, ow, oh, args.fps, args.out)
    info["video_total_fps_with_encode"] = round(len(path) / secs, 1)
    info["video"] = args.out
    print(json.dumps(info, indent=2))
    with open(os.path.splitext(args.out)[0] + "_stats.json", "w") as f:
        json.dump(info, f, indent=2)


if __name__ == "__main__":
    main()
