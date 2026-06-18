#!/usr/bin/env python3
"""Demo 3 figure: 'the model decides to refuse before it shows it'.

Recoverable-before-readable on Qwen3.5-2B refusal behaviour. A linear monitor on
the residual stream (recoverable) reaches high AUC several layers before the same
decision is readable from the logit lens (readable). Reads the inflight-safety run.
"""
from __future__ import annotations

import json
from pathlib import Path

import _style as st
import matplotlib.pyplot as plt

RUN = st.RUNS / "2026-06-17_qwen35_2b_safety_inflight"


def first_reach(xs, ys, target):
    for x, y in zip(xs, ys):
        if y >= target:
            return x
    return None


def main() -> None:
    st.apply_style()
    d = st.read_csv(RUN / "auc_by_depth.csv")
    summ = json.loads((RUN / "summary.json").read_text())
    layer = [int(float(v)) for v in d["layer"]]
    rec = st.as_float(d["recoverable_auc"])
    read = st.as_float(d["readable_auc"])
    target = float(summ["auc_target"])
    l_rec = summ["recoverable_reaches_target_at_layer"]
    l_read = summ["readable_reaches_target_at_layer"]
    gap = summ["recoverable_before_readable_gap_layers"]

    fig, ax = plt.subplots(figsize=(8.6, 5.2))

    # the gap band
    ax.axvspan(l_rec, l_read, color=st.VIOLET_FILL, alpha=0.9, zorder=0)

    ax.plot(layer, rec, "-o", ms=4.8, lw=2.8, color=st.BLUE,
            markeredgecolor=st.PAGE_BG, markeredgewidth=0.8,
            label="Recoverable: linear monitor on residual stream $h_\\ell$")
    ax.plot(layer, read, "-s", ms=4.0, lw=2.4, color=st.CORAL,
            label="Readable: same decision via the logit lens")

    ax.axhline(target, color=st.SLATE, ls="--", lw=1.1, alpha=0.8)
    ax.text(0.3, target + 0.012, f"AUC target = {target:g}", fontsize=9, color=st.SLATE)

    for lx, col, lab in [(l_rec, st.BLUE, "decodable\n@ layer %d" % l_rec),
                         (l_read, st.ORANGE, "visible in output\n@ layer %d" % l_read)]:
        ax.axvline(lx, color=col, ls=":", lw=1.4, alpha=0.85)
    ax.scatter([l_rec], [target], s=120, color=st.BLUE, zorder=5, edgecolors=st.PAGE_BG, linewidths=1.4)
    ax.scatter([l_read], [target], s=120, color=st.ORANGE, zorder=5, edgecolors=st.PAGE_BG, linewidths=1.4)

    mid = (l_rec + l_read) / 2
    ax.annotate(f"{gap}-layer gap\n({gap / summ['num_layers'] * 100:.0f}% of depth)",
                xy=(mid, target), xytext=(mid, target - 0.20),
                ha="center", va="top", fontsize=11, fontweight="bold", color=st.VIOLET_DK,
                arrowprops=dict(arrowstyle="-", color=st.VIOLET_DK, lw=0))

    ax.set_xlabel("Transformer layer $\\ell$")
    ax.set_ylabel("Refuse-vs-comply AUC")
    ax.set_xlim(0, summ["num_layers"])
    ax.set_ylim(0, 1.02)
    ax.set_title(
        f"Qwen3.5-2B \u00b7 {summ['n_refuse']} refusals vs {summ['n_comply']} compliances "
        "\u00b7 JailbreakBench prompts",
        fontsize=10, fontweight="normal", color=st.SLATE, pad=8)
    fig.suptitle("The model decides to refuse before it shows it",
                 fontsize=15, fontweight="bold", y=0.99)
    ax.legend(loc="lower right")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = st.save(fig, "fig3_recoverable_before_readable")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
