"""Run an experiment directly from the project checkout."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from tpu_gym.train import main

if __name__ == "__main__":
    main()
