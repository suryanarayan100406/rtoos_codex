"""Execute with: python scripts/kaggle_run.py --settings <yaml>."""

import sys
from pathlib import Path

# Support an unpacked source checkout as well as an installed package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drishti.runtime.kaggle import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
