"""Entry point: ``python -m disk_recorder`` opens the Tkinter window."""

from __future__ import annotations

import sys

from .gui import main

if __name__ == "__main__":
    sys.exit(main())
