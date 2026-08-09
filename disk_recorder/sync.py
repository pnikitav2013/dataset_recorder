"""Cross-correlation alignment of a capture to its played reference.

The board streams continuously and ships data in ~0.85 s bursts, so the exact
sample that corresponds to playback start is unknown from timing alone. We
therefore record a generous window and locate the reference inside it via
cross-correlation. This absorbs speaker/acoustic/mic/UART latency and yields
sub-100 ms start alignment regardless of buffering.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger("disk_recorder.sync")


@dataclass
class AlignResult:
    pcm: np.ndarray         # int16 mono, aligned and trimmed
    lag_samples: int        # offset of the reference within the capture
    correlation: float      # normalised peak correlation (0..1), confidence
    underrun: bool          # capture too short to cover reference + margin


def _normalise(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    x = x - x.mean()
    norm = np.linalg.norm(x)
    return x / norm if norm > 0 else x


def align(captured: np.ndarray, reference: np.ndarray, sample_rate: int,
          target_samples: int, extra_samples: int) -> AlignResult:
    """Locate ``reference`` inside ``captured`` and trim to the aligned clip.

    Both arrays must be mono at ``sample_rate``. ``captured`` is int16 (board
    PCM); ``reference`` is float. The returned clip has length
    ``target_samples + extra_samples`` (capped to what is available), keeping
    the allowed length increase within the configured margin.
    """
    from scipy.signal import correlate

    cap = captured.astype(np.float64)
    ref = reference.astype(np.float64)
    if cap.size == 0 or ref.size == 0:
        return AlignResult(pcm=captured[:0].astype(np.int16), lag_samples=0,
                           correlation=0.0, underrun=True)

    cap_n = _normalise(cap)
    ref_n = _normalise(ref)
    corr = correlate(cap_n, ref_n, mode="full", method="fft")
    peak_index = int(np.argmax(corr))
    lag = peak_index - (ref.size - 1)
    start = max(0, lag)

    want = target_samples + extra_samples
    segment = captured[start:start + want].astype(np.int16)
    underrun = segment.size < target_samples

    # Confidence: correlation between the reference and the aligned segment.
    compare_len = min(ref.size, segment.size)
    if compare_len > 1:
        a = _normalise(segment[:compare_len].astype(np.float64))
        b = ref_n[:compare_len]
        score = float(abs(np.dot(a, b)))
    else:
        score = 0.0

    logger.debug("align lag=%d corr=%.3f underrun=%s", lag, score, underrun)
    return AlignResult(pcm=segment, lag_samples=lag, correlation=score, underrun=underrun)
