"""Run gpu_render.py on a Kaggle T4 and download its outputs.

Bundles gpu_render.py with fixed command-line arguments into a private script
kernel, pushes it with the T4 accelerator, waits for it to finish and pulls
/kaggle/working into --out-dir. Needs the Kaggle CLI to be authenticated.

Files passed with --data are uploaded as a private Kaggle dataset (created on
first use, versioned after) and mounted in the kernel. In the render arguments,
write "@data/<file name>" to refer to them.

  python3 kaggle_run.py --slug shard-render-bench --out-dir bench -- bench
  python3 kaggle_run.py --slug yak-pan --out-dir out --data a.png --data b.png -- \
      render --shards '[{"path":"@data/a.png","yaw":0}]' --out /kaggle/working/pan.mp4
"""
import argparse
import json
import os
import shutil
import subprocess
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def sh(*cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(f"{' '.join(cmd)} failed:\n{r.stdout}\n{r.stderr}")
    return r.stdout


MARKER = "_shard_data_marker"

PRELUDE = f'''
import glob as _glob, os as _os
def _data_dir():
    hits = _glob.glob("/kaggle/input/**/{MARKER}", recursive=True)
    return _os.path.dirname(hits[0]) if hits else None
'''


def upload_dataset(user, slug, files):
    """Create or version a private dataset holding `files`; wait until it is ready."""
    ref = f"{user}/{slug}"
    work = tempfile.mkdtemp()
    for f in files:
        shutil.copy(f, os.path.join(work, os.path.basename(f)))
    open(os.path.join(work, MARKER), "w").close()
    with open(os.path.join(work, "dataset-metadata.json"), "w") as f:
        json.dump({"title": slug, "id": ref, "licenses": [{"name": "CC0-1.0"}]}, f)
    exists = subprocess.run(["kaggle", "datasets", "status", ref],
                            capture_output=True, text=True).returncode == 0
    if exists:
        print(sh("kaggle", "datasets", "version", "-p", work, "-m", "update", "-q").strip())
    else:
        print(sh("kaggle", "datasets", "create", "-p", work, "-q").strip())
    while True:
        r = subprocess.run(["kaggle", "datasets", "status", ref], capture_output=True, text=True)
        if "ready" in r.stdout.lower():
            return ref
        if "error" in r.stdout.lower():
            raise SystemExit(f"dataset {ref} failed: {r.stdout}")
        time.sleep(10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--accelerator", default="NvidiaTeslaT4")
    ap.add_argument("--dataset", action="append", default=[], help="owner/dataset to mount")
    ap.add_argument("--data", action="append", default=[], help="local file to upload and mount")
    ap.add_argument("render_args", nargs=argparse.REMAINDER)
    args = ap.parse_args()
    render_args = [a for a in args.render_args if a != "--"]

    out = sh("kaggle", "kernels", "list", "--mine", "--page-size", "1", "--csv")
    user = out.splitlines()[1].split(",")[0].split("/")[0]

    datasets = list(args.dataset)
    if args.data:
        datasets.append(upload_dataset(user, f"{args.slug}-data", args.data))

    src = open(os.path.join(HERE, "gpu_render.py")).read()
    src = src.replace('if __name__ == "__main__":\n    main()',
                      PRELUDE + "\nif __name__ == \"__main__\":\n    import sys\n"
                      f"    _args = {render_args!r}\n"
                      "    _d = _data_dir()\n"
                      "    if _d:\n"
                      "        _args = [a.replace('@data/', _d + '/') for a in _args]\n"
                      "    sys.argv = ['gpu_render.py'] + _args\n    main()")
    work = tempfile.mkdtemp()
    with open(os.path.join(work, "kernel.py"), "w") as f:
        f.write(src)
    meta = {
        "id": f"{user}/{args.slug}", "title": args.slug, "code_file": "kernel.py",
        "language": "python", "kernel_type": "script", "is_private": True,
        "enable_gpu": True, "enable_internet": False, "machine_shape": args.accelerator,
        "dataset_sources": datasets, "competition_sources": [], "kernel_sources": [],
    }
    with open(os.path.join(work, "kernel-metadata.json"), "w") as f:
        json.dump(meta, f)
    print(sh("kaggle", "kernels", "push", "-p", work, "--accelerator", args.accelerator).strip())

    ref = f"{user}/{args.slug}"
    t0 = time.time()
    while True:
        status = sh("kaggle", "kernels", "status", ref).strip()
        if any(s in status.lower() for s in ("complete", "error", "cancel")):
            break
        time.sleep(15)
    print(f"{status} after {time.time() - t0:.0f}s")
    os.makedirs(args.out_dir, exist_ok=True)
    print(sh("kaggle", "kernels", "output", ref, "-p", args.out_dir).strip())


if __name__ == "__main__":
    main()
