#!/usr/bin/env python3
"""Demo 2 figure: 'probes that cheat' — shortcut-control gates on CounterFact.

A shallow residual-stream probe looks like early factual recall, but a content-blind
frequency prior explains much of the shallow margin. Successive gates strip priors;
only the twin-reciprocal gate drives the content-blind prior to exactly zero -- and a
smaller, genuine depth-rising signal survives.

Panel A: share of the shallow (layer-0) margin explained by a content-blind prior,
         across the four gates (lower = cleaner).
Panel B: under the clean twin gate, the content-controlled probe margin still rises
         with depth while the content-blind prior stays flat at zero.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import _style as st
import matplotlib.pyplot as plt

VARIANTS = [
    ("counterfact", "Original\nCounterFact"),
    ("permute", "Global\npermutation"),
    ("balanced", "Within-relation\nbalancing"),
    ("twin", "Twin\nreciprocal"),
]

PRESETS = {
    "qwen35-2b": ("Qwen3.5-2B", {
        "counterfact": "factual_recall/runs/2026-06-17_qwen35_2b_counterfact",
        "permute": "factual_recall/runs/2026-06-17_qwen35_2b_counterfact_permute",
        "balanced": "factual_recall/runs/2026-06-17_qwen35_2b_counterfact_balanced",
        "twin": "factual_recall/runs/2026-06-17_qwen35_2b_counterfact_twin",
    }),
    "gpt2-medium": ("GPT-2 medium", {
        "counterfact": "factual_recall/runs/2026-06-04_gpt2med_counterfact",
        "permute": "factual_recall/runs/2026-06-04_gpt2med_counterfact_permute",
        "balanced": "factual_recall/runs/2026-06-04_gpt2med_counterfact_balanced",
        "twin": "factual_recall/runs/2026-06-04_gpt2med_counterfact_twin",
    }),
}


def prior_gate(run_dir: Path) -> dict:
    for name in ("counterfact_fulldepth_prior_gate.json", "counterfact_prior_gate.json"):
        p = run_dir / "outputs" / name
        if p.exists():
            return json.loads(p.read_text())
    raise FileNotFoundError(run_dir)


def depth_row(depths, wanted):
    return min(depths, key=lambda r: abs(int(r["depth"]) - wanted))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(PRESETS), default="qwen35-2b")
    args = ap.parse_args()
    st.apply_style()
    title_model, runs = PRESETS[args.model]

    prior_share, gates = [], []
    for key, label in VARIANTS:
        d = prior_gate(st.ROOT / runs[key])
        r0 = depth_row(d["depths"], 0)
        prior_share.append(max(0.0, float(r0["prior_fraction"])))
        gates.append(label)

    twin = prior_gate(st.ROOT / runs["twin"])
    depths = sorted(twin["depths"], key=lambda r: int(r["depth"]))
    layer = [int(r["depth"]) for r in depths]
    grounded = [float(r["grounded_margin_mean"]) for r in depths]
    prior_m = [float(r["prior_margin_mean"]) for r in depths]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.4, 5.2),
                                   gridspec_kw={"width_ratios": [1.05, 1.2]})

    # --- Panel A: cheat-meter (contaminated gates muted; only clean gate pops) --- #
    clean_idx = len(gates) - 1  # twin gate
    bar_colors = [st.BLUE if i == clean_idx else st.MUTED for i in range(len(gates))]
    bars = axA.bar(range(len(gates)), prior_share, color=bar_colors,
                   width=0.66, edgecolor=st.PAGE_BG, linewidth=1.5)
    for i, (b, v) in enumerate(zip(bars, prior_share)):
        axA.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.0%}",
                 ha="center", va="bottom", fontsize=11, fontweight="bold",
                 color=st.BLUE if i == clean_idx else st.SLATE)
    axA.set_xticks(range(len(gates)))
    axA.set_xticklabels(gates, fontsize=9)
    axA.set_ylabel("Share of shallow margin from\na content-blind frequency prior")
    axA.set_ylim(0, max(prior_share) * 1.25 + 0.05)
    axA.set_title("How much of 'early recall' is just a prior?", fontsize=11)
    axA.annotate("only this gate\nis clean", xy=(clean_idx, prior_share[clean_idx]),
                 xytext=(clean_idx - 0.45, max(prior_share) * 0.7), fontsize=9.5,
                 fontweight="bold", color=st.BLUE, ha="center",
                 arrowprops=dict(arrowstyle="->", color=st.BLUE, lw=1.4))

    # --- Panel B: surviving real effect under the clean gate --- #
    axB.axhline(0, color=st.SLATE, ls=":", lw=0.9, alpha=0.6)
    axB.plot(layer, grounded, "-o", ms=4.4, lw=2.8, color=st.BLUE,
             markeredgecolor=st.PAGE_BG, markeredgewidth=0.8,
             label="Content-controlled probe signal (survives)")
    axB.plot(layer, prior_m, "-s", ms=3.6, lw=2.0, color=st.GREY,
             label="Content-blind prior (zeroed by the gate)")
    axB.set_xlabel("Transformer layer (depth)")
    axB.set_ylabel("Recall margin (nats)")
    axB.set_title("Under the clean twin gate, a smaller real effect survives", fontsize=11)
    axB.legend(loc="center right", fontsize=9, borderpad=0.6)

    fig.suptitle(f"Probes that cheat: shortcut-control gates on {title_model}",
                 fontsize=15, fontweight="bold", y=0.99)
    fig.text(0.5, 0.925, "CounterFact factual recall \u00b7 endpoint-flow probe \u00b7 "
             "gates remove frequency / distractor / selection priors",
             ha="center", fontsize=9.5, color=st.SLATE)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out = st.save(fig, f"fig2_probes_that_cheat_{args.model}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
