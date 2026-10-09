"""Run gpu_render.py on a Kaggle T4 and download its outputs.

Bundles gpu_render.py with fixed command-line arguments into a private script
kernel, pushes it with the T4 accelerator, waits for it to finish and pulls
/kaggle/working into --out-dir. Needs the Kaggle CLI to be authenticated.

  python3 kaggle_run.py --slug shard-render-bench --out-dir ../renders/bench -- bench
"""
import argparse
import json
import os
import subprocess
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def sh(*cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(f"{' '.join(cmd)} failed:\n{r.stdout}\n{r.stderr}")
    return r.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--accelerator", default="NvidiaTeslaT4")
    ap.add_argument("--dataset", action="append", default=[], help="owner/dataset to mount")
    ap.add_argument("render_args", nargs=argparse.REMAINDER)
    args = ap.parse_args()
    render_args = [a for a in args.render_args if a != "--"]

    out = sh("kaggle", "kernels", "list", "--mine", "--page-size", "1", "--csv")
    user = out.splitlines()[1].split(",")[0].split("/")[0]

    src = open(os.path.join(HERE, "gpu_render.py")).read()
    src = src.replace('if __name__ == "__main__":\n    main()',
                      "if __name__ == \"__main__\":\n    import sys\n"
                      f"    sys.argv = ['gpu_render.py'] + {render_args!r}\n    main()")
    work = tempfile.mkdtemp()
    with open(os.path.join(work, "kernel.py"), "w") as f:
        f.write(src)
    meta = {
        "id": f"{user}/{args.slug}", "title": args.slug, "code_file": "kernel.py",
        "language": "python", "kernel_type": "script", "is_private": True,
        "enable_gpu": True, "enable_internet": False, "machine_shape": args.accelerator,
        "dataset_sources": args.dataset, "competition_sources": [], "kernel_sources": [],
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
