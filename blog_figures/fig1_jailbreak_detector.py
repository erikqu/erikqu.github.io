#!/usr/bin/env python3
"""Demo 1 figure: unsupervised jailbreak-success detector.

The endpoint probe's reconstruction residual ||h_L - hat h_L(l)|| flags successful
jailbreaks at shallow/mid depth, where Mahalanobis-to-clean-manifold is still near
chance. No labelled jailbreak-success data is used. Reads the anomaly-residual run.
"""
from __future__ import annotations

import json
from pathlib import Path

import _style as st
import matplotlib.pyplot as plt

RUN = st.RUNS / "2026-06-17_qwen35_2b_safety_anomaly"


def main() -> None:
    st.apply_style()
    d = st.read_csv(RUN / "anomaly_residual_auc.csv")
    summ = json.loads((RUN / "anomaly_residual_summary.json").read_text())
    layer = [int(float(v)) for v in d["layer"]]
    res = st.as_float(d["residual_strat"])
    maha = st.as_float(d["maha_strat"])
    maha_final = float(summ["maha_final"]["strat"])
    best = summ["best_residual_strat"]
    best_l, best_auc = int(best["layer"]), float(best["auc"])

    thr = 0.85

    def earliest(series):
        for lay, v in zip(layer, series):
            if v >= thr:
                return lay
        return None

    e_res, e_maha = earliest(res), earliest(maha)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.4, 5.2),
                                   gridspec_kw={"width_ratios": [2.5, 1]})

    # --- Panel A: AUC vs depth --- #
    axL.fill_between(layer, maha, res, where=[r > m for r, m in zip(res, maha)],
                     color=st.BLUE_FILL, alpha=0.8, zorder=0,
                     label="early-detection advantage")
    axL.plot(layer, res, "-o", ms=4.6, lw=2.8, color=st.BLUE,
             markeredgecolor=st.PAGE_BG, markeredgewidth=0.8,
             label="Endpoint residual (ours)")
    axL.plot(layer, maha, "-s", ms=3.8, lw=2.2, color=st.CORAL,
             label="Mahalanobis baseline")
    axL.axhline(thr, color=st.SLATE, ls="--", lw=1.1, alpha=0.8)
    axL.text(max(layer), thr + 0.012, f"AUC = {thr:g}", fontsize=9, color=st.SLATE, ha="right")
    axL.axhline(0.5, color=st.SLATE, ls=":", lw=0.9, alpha=0.6)
    for lx, col in [(e_res, st.BLUE), (e_maha, st.ORANGE)]:
        axL.scatter([lx], [thr], s=130, color=col, zorder=6, edgecolors=st.PAGE_BG, linewidths=1.5)
    axL.set_xlabel("Readout layer $\\ell$")
    axL.set_ylabel("Family-stratified success AUC")
    axL.set_xlim(0, max(layer))
    axL.set_ylim(0.35, 1.0)
    axL.set_title("The residual flags success early; the distance baseline catches up late",
                  fontsize=10.5)
    axL.legend(loc="lower right", fontsize=8.8, labelspacing=0.4, borderpad=0.6)

    # --- Panel B: earliest-detection bars (lower = better) --- #
    labels = ["Endpoint\nresidual\n(ours)", "Mahalanobis\nto clean\nmanifold"]
    vals = [e_res, e_maha]
    colors = [st.BLUE, st.CORAL]
    bars = axR.bar(labels, vals, color=colors, width=0.6, edgecolor=st.PAGE_BG, linewidth=1.5)
    for b, v in zip(bars, vals):
        axR.text(b.get_x() + b.get_width() / 2, v + 0.3, f"layer {v}",
                 ha="center", va="bottom", fontsize=11, fontweight="bold")
    axR.set_ylim(0, max(layer))
    axR.set_ylabel("Earliest layer reaching AUC 0.85")
    axR.set_title(f"{e_maha - e_res} layers earlier", fontsize=11)
    axR.annotate("", xy=(0, e_res + 0.5), xytext=(0, e_maha),
                 arrowprops=dict(arrowstyle="<->", color=st.SLATE, lw=1.3))

    fig.suptitle("Unsupervised jailbreak-success detection on Qwen3.5-2B",
                 fontsize=15, fontweight="bold", y=0.99)
    n = f"{summ['n_harmful']} harmful prompts \u00b7 {summ['n_success']} successful jailbreaks " \
        f"vs {summ['n_refusal']} refusals \u00b7 no labelled positives \u00b7 " \
        f"peak AUC comparable (residual {best_auc:.2f} vs Mahalanobis {float(summ['best_maha_layer_strat']['auc']):.2f})"
    fig.text(0.5, 0.925, n, ha="center", fontsize=9.0, color=st.SLATE)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out = st.save(fig, "fig1_jailbreak_detector")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
