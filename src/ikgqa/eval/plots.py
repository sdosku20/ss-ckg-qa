"""
ikgqa.eval.plots
================

Figures for the thesis, built from the result files in ``experiments/results``.

Every function here takes a DataFrame and returns a matplotlib ``Figure``. None
of them reads a file or writes one, so each can be exercised in a test without
touching the filesystem; ``experiments/figures.py`` does the reading and writing.

Three constraints shape the styling, and they are not decoration:

**The medium is print.** There is no hover layer and no dark mode, so every
series is identified three times over: by colour, by line style, and by marker.
A reader with a monochrome printer or with colour vision deficiency loses none of
the content. This is also why the yellow categorical slot is not used; at two
points on white paper it is illegible, and a figure nobody can read is worse than
a table.

**The x axis spans three orders of magnitude.** Retrieved subgraphs run from 8
nodes to 15,810, so size is plotted logarithmically. On a linear axis every
interesting configuration would collapse against the left edge.

**Size is the measured node count, never the parameter.** PCST's dial is a cost
per edge, the baselines' is a count or a hop number, and plotting recall against
"the parameter" would compare a cost with a count. Only measured size puts the
strategies on one axis.

Colours are the first, second, third, seventh and eighth slots of the reference
categorical order, which validates on the adjacent pairlist in light mode. The
fourth and fifth slots (yellow, magenta) are skipped for the print-contrast
reason above rather than reordered arbitrarily.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import pandas as pd

# Colour, line style and marker travel together, so identity never rests on
# colour alone. Order is fixed: a strategy keeps its appearance across every
# figure in the thesis, and filtering the set never repaints the survivors.
SERIES_STYLE: Dict[str, dict] = {
    "PCST": dict(color="#2a78d6", linestyle="-", marker="o"),
    "Top-k triples": dict(color="#eb6834", linestyle="--", marker="s"),
    "Top-k nodes": dict(color="#1baf7a", linestyle="-.", marker="^"),
    "BFS expansion": dict(color="#4a3aa7", linestyle=":", marker="D"),
    "Shortest paths": dict(color="#e34948", linestyle=(0, (3, 1, 1, 1)), marker="v"),
}

FALLBACK_STYLE = dict(color="#6b6b6b", linestyle="-", marker=".")

INK_PRIMARY = "#1a1a19"
INK_SECONDARY = "#5c5c58"
GRID = "#d8d8d4"


def style_for(name: str) -> dict:
    """Appearance for a series, matched leniently on the retriever's name.

    Result files name retrievers slightly differently across experiments
    ("Top-k triples (KAPING)" in one, "Top-k triples" in another). Matching on a
    prefix keeps one appearance per strategy without forcing the experiment
    scripts to agree on a label.
    """
    for key, style in SERIES_STYLE.items():
        if name.lower().startswith(key.lower()):
            return style
    return FALLBACK_STYLE


def _apply_axes_style(ax) -> None:
    """Recede the frame so the data carries the figure."""
    ax.grid(True, which="major", color=GRID, linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SECONDARY, labelsize=8)
    ax.xaxis.label.set_color(INK_PRIMARY)
    ax.yaxis.label.set_color(INK_PRIMARY)


def identical_recall_cluster(sweep: pd.DataFrame, places: int = 3):
    """The recall value the most strategies score identically, and where.

    Returns ``(size, recall, how_many_strategies)`` or ``None``. This is the
    chapter's central negative result: several strategies, connected and
    unconnected alike, land on exactly the same recall at the same size, which
    means the structural strategy is not what decides the outcome.

    Exact agreement is looked for rather than a tight spread, because those are
    different claims. A narrow spread says the strategies are similar; identical
    values to three decimal places say they retrieved the same answer nodes, and
    only the second is worth pointing at. Deriving it from the data keeps the
    annotation truthful if a later sweep moves the number, instead of freezing
    today's figure into the plot.
    """
    if sweep.empty:
        return None

    scored = sweep.assign(_bucket=sweep["answer_node_recall"].round(places))
    best, best_rank = None, None
    for value, group in scored.groupby("_bucket"):
        strategies = group["retriever"].nunique()
        if strategies < 2:
            continue
        # Ties on strategy count are broken by how many configurations landed on
        # the value. Two strategies agreeing once is a coincidence; eight
        # configurations across four strategies agreeing is the effect.
        rank = (strategies, len(group))
        if best_rank is None or rank > best_rank:
            best_rank = rank
            best = (float(group["nodes"].median()), float(value), int(strategies))
    return best


def recall_vs_size(sweep: pd.DataFrame, figsize=(6.0, 3.6)):
    """The headline figure: answer-node recall against measured subgraph size.

    One line per retrieval strategy, each swept across its own parameter range
    and plotted against the size it actually achieved. Reading a vertical slice
    answers the question the thesis asks: at this budget, which strategy recalls
    most?
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    for name, group in sweep.groupby("retriever", sort=False):
        points = group.sort_values("nodes")
        ax.plot(points["nodes"], points["answer_node_recall"],
                label=name, linewidth=1.6, markersize=5,
                markeredgecolor="white", markeredgewidth=0.6,
                **style_for(str(name)))

    cluster = identical_recall_cluster(sweep)
    if cluster is not None:
        size, recall, strategies = cluster
        ax.annotate(f"{strategies} strategies score exactly {recall:.3f}\n"
                    f"at ~{size:.0f} nodes",
                    xy=(size, recall), xytext=(size * 3.2, recall - 0.14),
                    fontsize=7.5, color=INK_SECONDARY,
                    arrowprops=dict(arrowstyle="-", color=INK_SECONDARY,
                                    linewidth=0.7, shrinkA=0, shrinkB=3))

    ax.set_xscale("log")
    ax.set_xlabel("Retrieved subgraph size (nodes, log scale)")
    ax.set_ylabel("Answer-node recall")
    ax.set_ylim(-0.03, 1.03)
    _apply_axes_style(ax)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_PRIMARY,
              loc="upper left", ncol=2)
    return fig


def _label_bars(ax, positions, values, offset=0.012) -> None:
    """Write each value at the end of its bar.

    Two reasons, and the second is the one that matters. Direct labels satisfy
    the relief rule for the lighter fills, which sit below 3:1 contrast on white.
    And a value of exactly zero draws no bar at all, so without a label the
    reader cannot tell a measured zero from a missing condition. On this data
    that distinction is a finding: the wrong-language catalogue really does score
    0.00, and it must not look like absent data.
    """
    for position, value in zip(positions, values):
        if pd.isna(value):
            continue
        ax.text(float(value) + offset, position, f"{float(value):.2f}",
                va="center", ha="left", fontsize=7, color=INK_SECONDARY)


def _apply_horizontal_style(ax) -> None:
    """Bars run horizontally so that long condition names stay readable.

    Condition labels here are phrases, not words. Rotating them or letting them
    collide on a vertical axis is the usual outcome; putting the categories on
    the y axis lets them be read straight.
    """
    _apply_axes_style(ax)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, 1.14)


def recall_and_ceiling(candidates: pd.DataFrame, figsize=(6.2, 3.4)):
    """What candidate generation kept, beside what the retriever then found.

    Plotting recall without its ceiling is what makes this kind of experiment
    misleading: a low score can mean the selector chose badly or that the region
    never contained the answer, and those need opposite fixes. The gap between
    the pair is the selector's shortfall; the ceiling itself is the generator's.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    summary = (candidates.groupby("condition", sort=False)[["recall", "ceiling"]]
               .mean().reset_index())
    summary = summary.iloc[::-1].reset_index(drop=True)   # first condition on top

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    positions = np.arange(len(summary))
    height = 0.36

    ax.barh(positions + height / 2 + 0.01, summary["ceiling"], height,
            label="Ceiling (kept by stage one)", color="#c9d9ef",
            edgecolor="white", linewidth=1.0)
    ax.barh(positions - height / 2 - 0.01, summary["recall"], height,
            label="Recall (found by the retriever)", color="#2a78d6",
            edgecolor="white", linewidth=1.0)

    _label_bars(ax, positions + height / 2 + 0.01, summary["ceiling"])
    _label_bars(ax, positions - height / 2 - 0.01, summary["recall"])

    ax.set_yticks(positions)
    ax.set_yticklabels(summary["condition"], fontsize=8)
    ax.set_xlabel("Fraction of answer nodes")
    _apply_horizontal_style(ax)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_PRIMARY,
              loc="lower right")
    return fig


def terminology_gain(gain: pd.DataFrame, figsize=(6.2, 3.4)):
    """Recall by catalogue condition, split by whether the diagnosis had a name.

    The split is the point. Named diagnoses are already retrievable and a
    catalogue can only hurt them; code-only diagnoses are unretrievable without
    one. An average over both hides each effect behind the other.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    table = (gain.groupby(["condition", "qkind"], sort=False)["answer_node_recall"]
             .mean().unstack("qkind"))
    table = table.iloc[::-1]                              # first condition on top

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    positions = np.arange(len(table.index))
    kinds = list(table.columns)
    height = 0.76 / max(len(kinds), 1)
    shades = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"]

    for offset, kind in enumerate(kinds):
        centred = positions + ((len(kinds) - 1) / 2 - offset) * (height + 0.01)
        ax.barh(centred, table[kind].values, height, label=str(kind),
                color=shades[offset % len(shades)], edgecolor="white", linewidth=1.0)
        _label_bars(ax, centred, table[kind].values)

    ax.set_yticks(positions)
    ax.set_yticklabels(table.index, fontsize=8)
    ax.set_xlabel("Answer-node recall")
    _apply_horizontal_style(ax)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_PRIMARY,
              loc="lower right")
    return fig


def size_reachable_by_pcst(sweep: pd.DataFrame, figsize=(6.0, 2.6)):
    """Subgraph size against PCST's own dial, on the sizes it can actually reach.

    Included because the sweep produced a finding that a recall curve conceals:
    varying the edge cost across its whole published range moved the result by
    three nodes. A method whose size dial does not control size is reporting a
    single point, whatever the parameter suggests.
    """
    import matplotlib.pyplot as plt

    pcst = sweep[sweep["retriever"].str.startswith("PCST")].sort_values("dial_value")

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    ax.plot(pcst["dial_value"], pcst["nodes"], linewidth=1.6, markersize=5,
            markeredgecolor="white", markeredgewidth=0.6, **style_for("PCST"))
    ax.set_xlabel(r"Edge cost $c_e$")
    ax.set_ylabel("Nodes returned")
    _apply_axes_style(ax)
    return fig


def save(fig, path, formats: Optional[Sequence[str]] = None) -> list:
    """Write a figure as vector PDF, plus PNG for quick viewing.

    PDF because LaTeX embeds it without resampling, so the text in the figure
    stays selectable and matches the body font weight at any zoom.
    """
    import pathlib

    formats = tuple(formats or ("pdf", "png"))
    base = pathlib.Path(path).with_suffix("")
    base.parent.mkdir(parents=True, exist_ok=True)

    written = []
    for suffix in formats:
        target = base.with_suffix("." + suffix)
        fig.savefig(target, format=suffix, dpi=200, bbox_inches="tight",
                    facecolor="white")
        written.append(target)
    return written
