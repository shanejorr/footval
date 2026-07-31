"""In-container runner: execute one candidate plot script, harvest figures.

Runs ONLY inside the footval-sandbox image — never on the host. Figures are
harvested even when the script raises, and anything the script saved itself is
copied out with a ``saved_`` prefix.
"""

import json
import os
import runpy
import shutil
import sys
import traceback

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".svg")


def main() -> int:
    script, outdir = sys.argv[1], sys.argv[2]
    workdir = "/tmp/work"
    os.makedirs(workdir, exist_ok=True)
    os.chdir(workdir)

    exception = None
    try:
        runpy.run_path(script, run_name="__main__")
    except BaseException as exc:  # candidate code: anything can happen
        exception = {"type": type(exc).__name__, "message": str(exc)[:2000]}
        traceback.print_exc()

    open_figures = []
    for num in plt.get_fignums():
        name = f"figure_{num:02d}.png"
        try:
            plt.figure(num).savefig(os.path.join(outdir, name), dpi=120)
            open_figures.append(name)
        except Exception:
            traceback.print_exc()

    saved_files = []
    for root, _dirs, files in os.walk(workdir):
        for fname in sorted(files):
            if fname.lower().endswith(IMAGE_EXTS):
                dest = "saved_" + fname
                try:
                    shutil.copy(os.path.join(root, fname), os.path.join(outdir, dest))
                    saved_files.append(dest)
                except Exception:
                    traceback.print_exc()

    manifest = {
        "open_figures": open_figures,
        "saved_files": saved_files,
        "exception": exception,
    }
    with open(os.path.join(outdir, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
    return 1 if exception else 0


if __name__ == "__main__":
    sys.exit(main())
