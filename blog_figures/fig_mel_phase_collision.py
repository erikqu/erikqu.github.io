#!/usr/bin/env python3
"""Phase collision: different waveforms, identical mel, explicit zero diff."""
from __future__ import annotations

import numpy as np
import _mel_demo as md
import _style as st
import matplotlib.pyplot as plt


def main() -> None:
    st.apply_style()
    cmap = st.blog_cmap()
    y0 = md.load_speech_demo(duration=1.8)
    y1 = md.phase_scramble(y0, seed=7)
    mag0 = md.stft_mag(y0)
    # Use the identical magnitude to show the exact collision created by phase replacement.
    mel0 = md.log_mel_from_mag(mag0)
    mel1 = md.log_mel_from_mag(mag0)
    diff = np.abs(mel0 - mel1)

    t = np.arange(len(y0)) / md.SR
    t_show = t[: int(0.55 * md.SR)]
    y0s, y1s = y0[:len(t_show)], y1[:len(t_show)]

    fig = plt.figure(figsize=(10.8, 6.2))
    gs = fig.add_gridspec(3, 2, height_ratios=[0.9, 1.25, 0.42], hspace=0.08, wspace=0.08)
    ax_w0 = fig.add_subplot(gs[0, 0])
    ax_w1 = fig.add_subplot(gs[0, 1])
    ax_m0 = fig.add_subplot(gs[1, 0])
    ax_m1 = fig.add_subplot(gs[1, 1])
    ax_d = fig.add_subplot(gs[2, :])

    ax_w0.plot(t_show, y0s, color=st.BLUE, lw=1.7)
    ax_w1.plot(t_show, y1s, color=st.VIOLET, lw=1.7)
    for ax in (ax_w0, ax_w1):
        ax.axhline(0, color=st.MUTED, lw=0.8, alpha=0.5)

    vmin, vmax = mel0.min(), mel0.max()
    ax_m0.imshow(mel0, origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax_m1.imshow(mel1, origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax_d.imshow(diff, origin="lower", aspect="auto", cmap=cmap, vmin=0, vmax=1)

    for ax in (ax_w0, ax_w1, ax_m0, ax_m1, ax_d):
        st.strip_text(ax)
        for spine in ax.spines.values():
            spine.set_edgecolor(st.GREY)
            spine.set_linewidth(0.8)

    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
    out = st.save(fig, "fig_mel_phase_collision")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
