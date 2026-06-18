"""Shared plotting style for the blog demo figures."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(exist_ok=True)

# ---- cool blue + violet palette ------------------------------------------ #
# Hero blue family (the "ours" series should always read as the star).
BLUE = "#2563eb"       # primary hero blue (ours)
BLUE_DK = "#1e3a8a"    # deep navy for emphasis text/markers
SKY = "#38bdf8"        # bright secondary blue
CYAN = "#06b6d4"
BLUE_FILL = "#dbeafe"  # light fill for advantage bands
SKY_FILL = "#e0f2fe"

# Secondary: violet -- an uncommon, elegant cool partner to blue (no warm tones).
VIOLET = "#7c3aed"     # baseline / contrast series
VIOLET_DK = "#5b21b6"  # deep violet for emphasis text
VIOLET_FILL = "#ede9fe"  # soft lavender band

# Neutral for de-emphasised elements (so highlighted items pop).
MUTED = "#b4c0d3"      # cool grey-blue

# Back-compat aliases (figures still reference these names).
CORAL = VIOLET
ORANGE = VIOLET
AMBER = VIOLET
GREEN = "#10b981"      # retained, unused
RED = "#ef4444"        # retained, unused

# Neutrals.
SLATE = "#475569"
GREY = "#94a3b8"
INK = "#0f172a"

# Blog page background -- figures render transparent and sit directly on this cream,
# so marker/bar "cut-out" edges use this color instead of white.
PAGE_BG = "#FCFCF5"


def apply_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 180,
        "savefig.dpi": 180,
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.edgecolor": "#cbd5e1",
        "axes.linewidth": 1.0,
        "axes.grid": True,
        "grid.color": "#e7e3d6",  # subtle warm grid that reads on the cream page
        "grid.linewidth": 0.9,
        "legend.frameon": False,
        "legend.fontsize": 9.5,
        "xtick.color": SLATE,
        "ytick.color": SLATE,
        "axes.labelcolor": INK,
        "text.color": INK,
        # transparent canvas so the blog's cream background shows through
        "figure.facecolor": "none",
        "axes.facecolor": "none",
        "savefig.facecolor": "none",
        "savefig.edgecolor": "none",
        "savefig.transparent": True,
    })


def save(fig, name: str):
    """Save a figure as a transparent SVG (and PNG fallback). Returns the SVG path."""
    for ext in ("svg", "png"):
        fig.savefig(OUT / f"{name}.{ext}", transparent=True, bbox_inches="tight")
    return OUT / f"{name}.svg"


def read_csv(path: Path) -> dict[str, list]:
    cols: dict[str, list] = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            for k, v in row.items():
                cols.setdefault(k, []).append(v)
    return cols


def as_float(xs: list) -> list[float]:
    return [float(x) for x in xs]


def blog_cmap():
    """Blue-to-violet spectrogram colormap matching the blog palette."""
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(
        "blog_mel",
        [BLUE_DK, BLUE, SKY, VIOLET, VIOLET_DK],
        N=256,
    )


def strip_text(ax, *, ticks: bool = False) -> None:
    """Remove all text from an axes — labels live in HTML captions."""
    ax.set_title("")
    ax.set_xlabel("")
    ax.set_ylabel("")
    if not ticks:
        ax.set_xticks([])
        ax.set_yticks([])
    ax.legend_ = None
