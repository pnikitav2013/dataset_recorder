"""channel_extractor — split one re-recorded channel into its own dataset.

A :mod:`disk_recorder` run stores every captured input device next to the
original clip as ``<name>_R_<prefix>.wav`` (see
:mod:`disk_recorder.storage`). A folder therefore mixes several sources, e.g.::

    116-288045-0024_R_fifine.wav
    116-288045-0024_R_X3800.wav
    116-288045-0024_R_mic1_stm32.wav
    116-288045.trans.txt

This tool copies such a dataset and keeps only one channel: it deletes audio
that does not carry the requested ``_R_<prefix>`` marker, then strips the marker
from the survivors so their names match the original clips again. Non-audio
files (transcripts, metadata) are copied verbatim. The extracted dataset root is
named ``<source_name>_R_<prefix>``.

Modules:

* :mod:`channel_extractor.extractor` — copy / filter / rename logic
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
