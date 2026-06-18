#!/usr/bin/env python3
"""Mel filterbank compression with separate linear and mel coordinate panels."""
from __future__ import annotations

import numpy as np
import _mel_demo as md
import _style as st
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def rounded(ax, x, y, w, h, fc, ec, lw=1.4):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.04",
                                facecolor=fc, edgecolor=ec, linewidth=lw))


def main() -> None:
    st.apply_style()
    E = md.mel_basis()
    freqs = np.linspace(0, 1, E.shape[1])
    y = md.load_speech_demo(duration=1.8)
    mag = md.stft_mag(y)
    frame = mag[:, int(np.argmax(np.sum(mag, axis=0)))]
    frame = frame / (frame.max() or 1)
    mel = E @ (frame**2)
    mel = mel / (mel.max() or 1)

    fig, ax = plt.subplots(figsize=(10.8, 5.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")

    # Top: filterbank over a common linear frequency axis.
    for m, a in zip(np.linspace(0, md.N_MELS - 1, 18, dtype=int), np.linspace(0.35, 1.0, 18)):
        xs = 0.55 + freqs * 8.9
        ys = 3.15 + E[m] / (E.max() or 1) * 1.35
        ax.plot(xs, ys, color=st.BLUE, alpha=a, lw=1.55)

    # Bottom: separate coordinate spaces, connected by one compression arrow.
    rounded(ax, 0.55, 0.55, 5.15, 1.65, "none", st.BLUE, lw=1.5)
    rounded(ax, 7.10, 0.55, 2.35, 1.65, "none", st.VIOLET, lw=1.7)
    lx = 0.75 + np.linspace(0, 4.75, len(frame))
    ly = 0.72 + frame * 1.25
    ax.fill_between(lx, 0.72, ly, color=st.BLUE_FILL, alpha=0.95)
    ax.plot(lx, ly, color=st.BLUE, lw=1.5)

    n = len(mel)
    bar_w = 2.05 / n
    for i, v in enumerate(mel):
        x = 7.25 + i * bar_w
        ax.add_patch(plt.Rectangle((x, 0.72), bar_w * 0.78, v * 1.25,
                                   facecolor=st.VIOLET, edgecolor="none", alpha=0.28 + 0.62 * v))

    ax.add_patch(FancyArrowPatch((5.92, 1.38), (6.92, 1.38), arrowstyle="-|>", mutation_scale=16,
                                 linewidth=2.5, color=st.VIOLET_DK, shrinkA=6, shrinkB=6))

    fig.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.02)
    out = st.save(fig, "fig_mel_filterbank")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
