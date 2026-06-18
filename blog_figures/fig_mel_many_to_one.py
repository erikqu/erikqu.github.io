#!/usr/bin/env python3
"""Many-to-one mel map with waveform and mel motifs, no text."""
from __future__ import annotations

import numpy as np
import _style as st
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def rounded(ax, x, y, w, h, fc, ec, lw=1.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.05",
                                facecolor=fc, edgecolor=ec, linewidth=lw))


def waveform(ax, x, y, w, h, color, phase=0, extra=0):
    t = np.linspace(0, 1, 160)
    sig = np.sin(2*np.pi*(3.6*t + phase)) + extra*np.sin(2*np.pi*10*t)
    sig /= max(abs(sig).max(), 1e-6)
    ax.plot(x+t*w, y+h*(0.5+0.34*sig), color=color, lw=1.9, solid_capstyle="round")


def mel_tile(ax, x, y, w, h):
    z = np.zeros((11, 15))
    z[2, :] = np.linspace(0.25, 0.95, 15)
    z[5, :] = np.linspace(0.45, 1.0, 15)
    z[8, :] = np.linspace(0.18, 0.75, 15)
    ax.imshow(z, extent=(x+0.08, x+w-0.08, y+0.08, y+h-0.08), origin="lower",
              cmap=st.blog_cmap(), aspect="auto", zorder=2)
    rounded(ax, x, y, w, h, "none", st.VIOLET, lw=2.2)


def arrow(ax, p0, p1, color, lw=2.2, dash=None, rad=0):
    a = FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=14, linewidth=lw,
                        color=color, shrinkA=8, shrinkB=8, connectionstyle=f"arc3,rad={rad}")
    if dash:
        a.set_linestyle((0, dash))
    ax.add_patch(a)


def main() -> None:
    st.apply_style()
    fig, ax = plt.subplots(figsize=(11, 4.7))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 5)
    ax.axis("off")

    mel_x, mel_y, mw, mh = 4.75, 2.0, 1.25, 1.0
    mel_tile(ax, mel_x, mel_y, mw, mh)

    left = [(0.55, 3.65, 0.0, 0.0), (0.55, 2.25, 0.18, 0.18), (0.55, 0.85, 0.33, 0.35)]
    for i, (x, y, phase, extra) in enumerate(left):
        rounded(ax, x, y, 1.25, 0.72, st.BLUE_FILL, st.BLUE, lw=1.8)
        waveform(ax, x+0.16, y+0.12, 0.93, 0.48, st.BLUE_DK, phase=phase, extra=extra)
        arrow(ax, (x+1.25, y+0.36), (mel_x, mel_y+mh/2), st.BLUE, lw=2.4, rad=0.16 if y>2.4 else -0.16 if y<1.5 else 0)

    right = [(8.95, 3.65, 0.05, 0.05), (8.95, 2.25, 0.27, 0.25), (8.95, 0.85, 0.48, 0.45)]
    for x, y, phase, extra in right:
        rounded(ax, x, y, 1.25, 0.72, st.BLUE_FILL, st.VIOLET, lw=1.8)
        waveform(ax, x+0.16, y+0.12, 0.93, 0.48, st.VIOLET_DK, phase=phase, extra=extra)
        arrow(ax, (mel_x+mw, mel_y+mh/2), (x, y+0.36), st.VIOLET, lw=2.3, dash=(6, 5), rad=-0.13 if y>2.4 else 0.13 if y<1.5 else 0)

    # halo around central collapse point
    for r, a in [(0.85, 0.12), (1.08, 0.08)]:
        ax.add_patch(plt.Circle((mel_x+mw/2, mel_y+mh/2), r, fill=False, edgecolor=st.VIOLET, linewidth=1.5, alpha=a))

    fig.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.02)
    out = st.save(fig, "fig_mel_many_to_one")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
