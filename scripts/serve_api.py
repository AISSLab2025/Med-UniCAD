"""FastAPI server entry point."""

import sys
from pathlib import Path


# Permit direct execution from a source checkout before an editable install.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from medunicad.inference.api import main


if __name__ == "__main__":
    main()
