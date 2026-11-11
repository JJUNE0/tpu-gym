"""Locate the JAX learner inside the pinned FlashCHORD submodule."""

from pathlib import Path
import sys


def register_flash_chord():
    root = Path(__file__).resolve().parents[2]
    source = root / "third_party/flash_chord/robotic_grounding/flash_chord/src"
    if not (source / "flash_chord/training/flash_sac/learner.py").is_file():
        raise RuntimeError("FlashCHORD is missing. Run: python3 scripts/bootstrap.py")
    if not (source / "flash_chord/lifecycle/reset_context.py").is_file():
        raise RuntimeError("FlashCHORD needs the local JAX patch. Run: python3 scripts/bootstrap.py")
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
