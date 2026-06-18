#!/usr/bin/env python3
"""Export demo WAV clips for the log-mel blog post."""
from __future__ import annotations

from pathlib import Path

import _mel_demo as md
import numpy as np
from scipy.io import wavfile

OUT = Path(__file__).resolve().parent / "outputs" / "audio"
OUT.mkdir(parents=True, exist_ok=True)


def save(name: str, y: np.ndarray) -> Path:
    y = np.clip(y, -1.0, 1.0)
    path = OUT / f"{name}.wav"
    wavfile.write(path, md.SR, (y * 32767).astype(np.int16))
    return path


def main() -> None:
    y_orig = md.load_speech_demo(duration=1.8)
    y_scram = md.phase_scramble(y_orig, seed=7)
    y_gl = md.classical_mel_roundtrip(y_orig)

    for name, y in [
        ("01_original", y_orig),
        ("02_phase_scrambled", y_scram),
        ("03_griffin_lim_roundtrip", y_gl),
    ]:
        p = save(name, y)
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
