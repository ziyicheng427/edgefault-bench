#!/usr/bin/env python3
"""Generate Figure 3 (verification + evaluation panels) from committed artefacts.

Sources:
  results/verification/mutation-matrix-2026-11-07.json
  results/verification/hust-lodo-outer-2026-11-07.json

Run from the repository root:
    uv run python paper/figures/plot_results.py
Outputs paper/figures/figure3.pdf (vector).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
MUTATION = ROOT / "results/verification/mutation-matrix-2026-11-07.json"
LODO = ROOT / "results/verification/hust-lodo-outer-2026-11-07.json"
OUT = Path(__file__).resolve().parent / "figure3.pdf"

# ColorBrewer Set2 (colour-blind safe).
SET2 = {
    "INV-1": "#66c2a5",
    "INV-2": "#fc8d62",
    "INV-3": "#8da0cb",
    "INV-4": "#e78ac3",
    "INV-5": "#a6d854",
    "INV-6": "#ffd92f",
}

mutation = json.loads(MUTATION.read_text())
mutations = mutation["mutations"]
mechanisms = sorted({entry["mechanism"] for entry in mutations})
invariant_order = [f"INV-{i}" for i in range(1, 7)]
rows = [m for inv in invariant_order for m in mutations if m["target_invariant"] == inv]
row_labels = [m["mutation_id"] for m in rows]
inv_colors = [SET2[m["target_invariant"]] for m in rows]

fig, (ax_left, ax_right) = plt.subplots(
    1, 2, figsize=(9.6, 4.4), gridspec_kw={"width_ratios": [1.35, 1.0]}
)

# Left: detection dot matrix.
for row_index, entry in enumerate(rows):
    column = mechanisms.index(entry["mechanism"])
    ax_left.scatter(
        column, row_index, s=130, marker="o",
        color=inv_colors[row_index], edgecolors="#333333", linewidths=0.6, zorder=3,
    )
    for other in range(len(mechanisms)):
        if other != column:
            ax_left.scatter(
                other, row_index, s=10, marker=".", color="#cccccc", zorder=1,
            )
ax_left.set_yticks(range(len(rows)))
ax_left.set_yticklabels(row_labels, fontsize=8)
ax_left.set_xticks(range(len(mechanisms)))
ax_left.set_xticklabels(
    [name.replace(" ", "\n", 1) for name in mechanisms],
    fontsize=6.2, rotation=55, rotation_mode="anchor", ha="right",
)
ax_left.set_xlim(-0.6, len(mechanisms) - 0.4)
ax_left.set_ylim(len(rows) - 0.4, -0.6)
ax_left.set_title(
    f"Fault injection: {mutation['detected']}/{mutation['total']} detected", fontsize=9
)
for spine in ("top", "right"):
    ax_left.spines[spine].set_visible(False)
for inv, color in SET2.items():
    ax_left.scatter([], [], color=color, label=inv)
ax_left.legend(
    fontsize=7, ncol=6, loc="upper center", bbox_to_anchor=(0.5, -0.42),
    frameon=False, columnspacing=1.0,
)

# Right: LODO folds with per-seed points and CIs.
lodo = json.loads(LODO.read_text())
folds = lodo["folds"]
order = sorted(folds)
means = [folds[fold]["mean_macro_f1"] for fold in order]
positions = range(len(order))
for position, fold in enumerate(order):
    seed_scores = [entry["macro_f1"] for entry in folds[fold]["seed_scores"]]
    ax_right.scatter(
        [position] * len(seed_scores), seed_scores,
        s=22, color="#8da0cb", edgecolors="#333333", linewidths=0.5, zorder=3,
        label="seed runs" if position == 0 else None,
    )
ax_right.plot(
    positions, means, "o--", color="#fc8d62", markersize=7, linewidth=1.2,
    label="fold mean", zorder=4,
)
outer = lodo["outer_mean_of_fold_means"]
ax_right.axhline(outer, color="#333333", linestyle=":", linewidth=1.0)
ax_right.text(
    len(order) - 0.45, outer + 0.015, f"outer mean {outer:.3f}",
    fontsize=7.5, ha="right",
)
ax_right.set_xticks(list(positions))
ax_right.set_xticklabels(order, fontsize=8)
ax_right.set_xlabel("unseen test device", fontsize=8)
ax_right.set_ylabel("test Macro-F1", fontsize=8)
ax_right.set_ylim(0.0, 0.85)
ax_right.set_title("Leave-one-device-out (audit policy)", fontsize=9)
ax_right.legend(fontsize=7, loc="upper right", frameon=False)
for spine in ("top", "right"):
    ax_right.spines[spine].set_visible(False)

fig.tight_layout()
fig.savefig(OUT)
print(f"wrote {OUT}")
