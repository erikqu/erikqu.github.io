#!/usr/bin/env python3
"""Text-free log-mel pipeline diagram with visual motifs."""
from __future__ import annotations

import numpy as np
import _style as st
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


def rounded(ax, x, y, w, h, fc, ec, lw=1.8):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.06",
                       facecolor=fc, edgecolor=ec, linewidth=lw)
    ax.add_patch(p)
    return p


def arrow(ax, start, end, color, lw=2.5, dash=None, rad=0):
    a = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14,
                        linewidth=lw, color=color, shrinkA=8, shrinkB=8,
                        connectionstyle=f"arc3,rad={rad}")
    if dash:
        a.set_linestyle((0, dash))
    ax.add_patch(a)


def waveform(ax, x, y, w, h, color, phase=0):
    t = np.linspace(0, 1, 160)
    sig = np.sin(2*np.pi*(4*t + phase)) * (0.35 + 0.45*np.sin(np.pi*t)**2)
    ax.plot(x + t*w, y + h*(0.5 + 0.35*sig), color=color, lw=2.0, solid_capstyle="round")


def mel_tile(ax, x, y, w, h, border=st.VIOLET):
    rng = np.random.default_rng(1)
    tile = np.zeros((9, 13))
    for band in [2, 4, 6]:
        tile[band, :] = np.linspace(0.3, 1.0, tile.shape[1])
    tile += 0.10*rng.random(tile.shape)
    ax.imshow(tile, extent=(x, x+w, y, y+h), origin="lower", cmap=st.blog_cmap(), aspect="auto", zorder=2)
    rounded(ax, x, y, w, h, "none", border, lw=2.0)


def spectrogram_icon(ax, x, y, w, h):
    for i, alpha in enumerate(np.linspace(0.18, 0.85, 7)):
        yy = y + h*(0.12 + i*0.12)
        ax.add_patch(Rectangle((x+w*0.1, yy), w*0.8, h*0.055, color=st.BLUE, alpha=alpha, linewidth=0))


def filter_icon(ax, x, y, w, h):
    for i, a in enumerate(np.linspace(0.35, 1.0, 5)):
        cx = x + w*(0.15 + i*0.17)
        ww = w*0.22
        ax.plot([cx-ww/2, cx, cx+ww/2], [y+h*0.18, y+h*0.82, y+h*0.18], color=st.BLUE, alpha=a, lw=1.8)


def log_icon(ax, x, y, w, h):
    t = np.linspace(0.05, 1, 100)
    ax.plot(x + t*w*0.9, y + h*(0.18 + 0.62*np.log1p(6*t)/np.log(7)), color=st.BLUE_DK, lw=2.0)


def main() -> None:
    st.apply_style()
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 3.6)
    ax.axis("off")

    xs = [0.35, 2.0, 3.65, 5.3, 6.95]
    w, h, y = 1.05, 0.82, 1.85
    fills = [st.SKY_FILL, st.BLUE_FILL, st.BLUE_FILL, st.BLUE_FILL, st.VIOLET_FILL]
    edges = [st.SKY, st.BLUE, st.BLUE, st.BLUE, st.VIOLET]
    for x, fc, ec in zip(xs, fills, edges):
        rounded(ax, x, y, w, h, fc, ec)
    waveform(ax, xs[0]+0.13, y+0.12, w-0.26, h-0.24, st.BLUE_DK)
    spectrogram_icon(ax, xs[1], y, w, h)
    filter_icon(ax, xs[2], y, w, h)
    log_icon(ax, xs[3]+0.08, y+0.12, w-0.16, h-0.24)
    mel_tile(ax, xs[4]+0.08, y+0.10, w-0.16, h-0.20)

    for i in range(len(xs)-1):
        arrow(ax, (xs[i]+w, y+h/2), (xs[i+1], y+h/2), st.BLUE, lw=2.6)

    # discarded phase/detail branches, no text
    arrow(ax, (xs[1]+w/2, y), (xs[1]+w/2, 0.65), st.VIOLET, lw=1.9, dash=(4, 4))
    for k in range(4):
        ax.plot(xs[1]+w/2 + 0.16*np.cos(k), 0.50 + 0.07*k, marker="o", ms=3.8, color=st.VIOLET, alpha=0.35+0.12*k)

    # lossy return path
    arrow(ax, (xs[-1]+w/2, y-0.05), (xs[0]+w/2, 0.55), st.VIOLET_DK, lw=2.6, dash=(7, 5), rad=-0.25)
    waveform(ax, xs[0]+0.25, 0.22, 0.62, 0.42, st.VIOLET_DK, phase=0.25)

    fig.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.02)
    out = st.save(fig, "fig_mel_pipeline")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
