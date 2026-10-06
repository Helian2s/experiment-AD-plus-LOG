"""Launch the current experiment; its code and results live under experiments/."""

from pathlib import Path
import runpy
import sys


if __name__ == "__main__":
    script = (
        Path(__file__).resolve().parent
        / "experiments"
        / "003_corrected_full"
        / "run_experiment.py"
    )
    sys.path.insert(0, str(script.parent))
    try:
        runpy.run_path(str(script), run_name="__main__")
    finally:
        sys.path.pop(0)
