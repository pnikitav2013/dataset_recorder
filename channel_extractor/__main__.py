"""Entry point: ``python -m channel_extractor <source> <prefix> <dest>``.

Extracts a single re-recorded channel from a :mod:`disk_recorder` dataset into a
new ``<source_name>_R_<prefix>`` folder under ``<dest>``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .extractor import extract


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="channel_extractor",
        description="Split one '_R_<prefix>' channel of a re-recorded dataset into its own dataset.",
    )
    parser.add_argument("source", type=Path, help="source re-recorded dataset folder")
    parser.add_argument("prefix", help="channel/source prefix to keep (the part after '_R_')")
    parser.add_argument(
        "dest",
        type=Path,
        help="destination parent dir; the dataset '<source_name>_R_<prefix>' is created inside it",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing extracted dataset instead of failing",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="log each kept/removed file")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    try:
        extract(args.source, args.prefix, args.dest, overwrite=args.overwrite)
    except (FileExistsError, NotADirectoryError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    sys.exit(main())
