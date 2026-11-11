"""Initialize pinned submodules, apply the JAX patch, and optionally fetch models."""

import argparse
import os
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[1]
FLASH_CHORD = Path("third_party/flash_chord")
MODULES = (
    Path("third_party/mujoco_playground"), FLASH_CHORD,
    Path("third_party/FlashSAC"), Path("third_party/sacpp"),
)


def run(*args, cwd=ROOT, **kwargs):
    env = {**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"}
    return subprocess.run(args, cwd=cwd, env=env, check=True, text=True, **kwargs)


def bootstrap(assets=None, skip_sacpp=False):
    # Configure sparse checkout before the first checkout of video_to_data.
    # Its LFS datasets and unrelated project folders are not needed by the learner.
    module = ROOT / FLASH_CHORD
    cloned = not (module / ".git").exists()
    if cloned:
        url = run("git", "config", "-f", ".gitmodules", "--get",
                  f"submodule.{FLASH_CHORD}.url", capture_output=True).stdout.strip()
        run("git", "clone", "--filter=blob:none", "--no-checkout", url, str(module))
    run("git", "sparse-checkout", "init", "--cone", cwd=module)
    run("git", "sparse-checkout", "set", "robotic_grounding/flash_chord", cwd=module)
    if cloned:
        # A --no-checkout clone can already have HEAD at the requested commit.
        # submodule update then skips checkout, so populate its first worktree.
        entry = run("git", "ls-files", "--stage", "--", str(FLASH_CHORD),
                    capture_output=True).stdout.split()
        if len(entry) < 2 or entry[0] != "160000":
            raise RuntimeError("FlashCHORD is missing its pinned gitlink entry")
        run("git", "checkout", "--detach", entry[1], cwd=module)
    modules = [path for path in MODULES if not (skip_sacpp and path.name == "sacpp")]
    run("git", "submodule", "update", "--init", "--filter=blob:none", "--",
        *(str(path) for path in modules))
    run("git", "submodule", "absorbgitdirs")

    patch = ROOT / "patches/flash_chord-jax.patch"
    applied = subprocess.run(
        ["git", "apply", "--reverse", "--check", str(patch)],
        cwd=module, capture_output=True,
    ).returncode == 0
    if not applied:
        run("git", "apply", "--check", str(patch), cwd=module)
        run("git", "apply", str(patch), cwd=module)
    print("FlashCHORD JAX patch ready.", flush=True)

    if assets:
        package = ROOT / "third_party/mujoco_playground/mujoco_playground"
        code = (package / "_src/mjx_env.py").read_text()
        commit = re.search(r'MENAGERIE_COMMIT_SHA\s*=\s*"([a-f0-9]+)"', code).group(1)
        target = package / "external_deps/mujoco_menagerie"
        if not (target / ".git").exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            run("git", "clone", "--filter=blob:none", "--no-checkout",
                "https://github.com/google-deepmind/mujoco_menagerie.git", str(target))
            run("git", "sparse-checkout", "init", "--cone", cwd=target)
        run("git", "sparse-checkout", "add", *assets, cwd=target)
        run("git", "checkout", "--detach", commit, cwd=target)
        print(f"Menagerie models ready: {', '.join(assets)}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", nargs="+", metavar="MODEL",
                        help="Sparse Menagerie models, e.g. unitree_g1 franka_emika_panda")
    parser.add_argument("--skip-sacpp", action="store_true",
                        help="Skip the private SAC++ reference submodule (PPO and FlashSAC still work)")
    args = parser.parse_args()
    bootstrap(args.assets, args.skip_sacpp)


if __name__ == "__main__":
    main()
