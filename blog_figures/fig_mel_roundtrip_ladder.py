#!/usr/bin/env python3
"""Classical vs neural decode paths with icon motifs and no text."""
from __future__ import annotations

import numpy as np
import _style as st
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle, Rectangle


def rounded(ax, x, y, w, h, fc, ec, lw=1.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.05",
                                facecolor=fc, edgecolor=ec, linewidth=lw))


def arrow(ax, p0, p1, color, lw=2.2, dash=None):
    a = FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=13, linewidth=lw,
                        color=color, shrinkA=7, shrinkB=7)
    if dash:
        a.set_linestyle((0, dash))
    ax.add_patch(a)


def mel_tile(ax, x, y, w, h, border=st.VIOLET):
    z = np.zeros((8, 10))
    z[[2, 4, 5], :] = np.linspace(0.25, 1, 10)
    ax.imshow(z, extent=(x+0.08, x+w-0.08, y+0.08, y+h-0.08), origin="lower",
              cmap=st.blog_cmap(), aspect="auto", zorder=2)
    rounded(ax, x, y, w, h, "none", border, lw=1.8)


def bars(ax, x, y, w, h, color=st.BLUE):
    vals = np.array([.15, .55, .25, .85, .35, .62, .2])
    bw = w / (len(vals) * 1.4)
    for i, v in enumerate(vals):
        ax.add_patch(Rectangle((x + i*bw*1.4, y), bw, h*v, color=color, alpha=0.28+0.55*v, linewidth=0))


def swirl(ax, x, y, w, h):
    theta = np.linspace(0, 4*np.pi, 180)
    r = np.linspace(0.04, 0.40, len(theta))
    ax.plot(x+w/2 + r*np.cos(theta)*w, y+h/2 + r*np.sin(theta)*h, color=st.VIOLET, lw=1.9)


def waveform(ax, x, y, w, h, color, noisy=False):
    t = np.linspace(0, 1, 180)
    sig = np.sin(2*np.pi*4*t)
    if noisy:
        sig += 0.35*np.sin(2*np.pi*17*t)
    ax.plot(x+t*w, y+h*(0.5+0.34*sig), color=color, lw=2.0, solid_capstyle="round")


def fan(ax, x, y, w, h):
    for i, yy in enumerate(np.linspace(0.18, 0.82, 5)):
        ax.plot([x+w*0.20, x+w*0.82], [y+h*0.5, y+h*yy], color=st.BLUE, alpha=0.45+0.1*i, lw=1.5)


def neural_stack(ax, x, y, w, h):
    for i in range(4):
        dx = i * w * 0.12
        ax.add_patch(FancyBboxPatch((x+dx, y+dx*0.18), w*0.62, h*0.72,
                                    boxstyle="round,pad=0.02,rounding_size=0.04",
                                    facecolor=st.BLUE_FILL, edgecolor=st.BLUE, linewidth=1.4, alpha=0.96))


def main() -> None:
    st.apply_style()
    fig, ax = plt.subplots(figsize=(11, 4.4))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 4.6)
    ax.axis("off")

    # Classical path: mel -> pseudo-inverse fan -> magnitude bars -> phase swirl -> noisy waveform
    y1 = 3.05
    xs = [0.35, 2.1, 3.85, 5.6, 7.35]
    for x in xs:
        rounded(ax, x, y1, 1.05, 0.75, st.BLUE_FILL, st.BLUE)
    mel_tile(ax, xs[0], y1, 1.05, 0.75)
    fan(ax, xs[1]+0.1, y1+0.1, 0.85, 0.55)
    bars(ax, xs[2]+0.16, y1+0.15, 0.75, 0.45)
    swirl(ax, xs[3]+0.12, y1+0.10, 0.82, 0.55)
    waveform(ax, xs[4]+0.12, y1+0.14, 0.82, 0.45, st.VIOLET_DK, noisy=True)
    for a, b in zip(xs[:-1], xs[1:]):
        arrow(ax, (a+1.05, y1+0.375), (b, y1+0.375), st.BLUE)

    # Neural path: mel -> stack -> smooth waveform
    y2 = 0.85
    rounded(ax, 0.35, y2, 1.05, 0.75, st.VIOLET_FILL, st.VIOLET)
    mel_tile(ax, 0.35, y2, 1.05, 0.75)
    rounded(ax, 2.65, y2, 1.4, 0.75, st.BLUE_FILL, st.BLUE, lw=1.8)
    neural_stack(ax, 2.78, y2+0.10, 1.1, 0.58)
    rounded(ax, 5.25, y2, 1.2, 0.75, st.BLUE_FILL, st.BLUE_DK, lw=2.0)
    waveform(ax, 5.42, y2+0.15, 0.86, 0.44, st.BLUE_DK, noisy=False)
    arrow(ax, (1.40, y2+0.375), (2.65, y2+0.375), st.VIOLET)
    arrow(ax, (4.05, y2+0.375), (5.25, y2+0.375), st.VIOLET)

    fig.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.02)
    out = st.save(fig, "fig_mel_roundtrip_ladder")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
