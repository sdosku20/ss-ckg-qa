"""
Figures for the thesis.

A figure that disagrees with the table beside it is worse than no figure, so
these tests check the properties a reader would otherwise have to take on trust:
that a strategy keeps one appearance everywhere, that size is plotted on a log
axis, that the annotated convergence is derived from the data rather than typed
in, and that a measured zero is visible as a zero.

matplotlib is an optional dependency (figures are built from committed CSVs, not
during a normal run), so the whole module skips when it is absent.
"""

from __future__ import annotations

import pandas as pd
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from ikgqa.eval import plots  # noqa: E402


@pytest.fixture
def sweep():
    """A sweep in the shape experiments/sweep_replica.py writes."""
    return pd.DataFrame([
        ("PCST", 0.5, 11.0, 0.164),
        ("PCST", 0.1, 11.0, 0.164),
        ("PCST", 0.01, 15810.0, 1.000),
        ("top-k triples (KAPING)", 10, 11.0, 0.164),
        ("top-k triples (KAPING)", 100, 93.6, 0.352),
        ("shortest paths", 10, 11.0, 0.164),
        ("shortest paths", 2, 3.0, 0.147),
        ("top-k nodes + neighbours", 10, 21.7, 0.164),
        ("BFS expansion", 1, 7.7, 0.150),
        ("BFS expansion", 2, 15333.7, 0.925),
    ], columns=["retriever", "dial_value", "nodes", "answer_node_recall"])


# --- the convergence annotation must come from the data ---------------------

def test_the_annotated_cluster_is_the_one_the_chapter_claims(sweep):
    size, recall, strategies = plots.identical_recall_cluster(sweep)
    assert recall == pytest.approx(0.164)
    assert strategies == 4, "PCST, KAPING, shortest paths and top-k nodes all tie"
    assert size == pytest.approx(11.0)


def test_more_configurations_break_a_tie_on_strategy_count():
    # Two clusters, four strategies each. The one with more configurations
    # landing on it is the effect; the other could be coincidence.
    frame = pd.DataFrame(
        [("a", 5.0, 0.50), ("b", 5.0, 0.50), ("c", 5.0, 0.50), ("d", 5.0, 0.50)]
        + [("a", 9.0, 0.90), ("b", 9.0, 0.90), ("c", 9.0, 0.90), ("d", 9.0, 0.90),
           ("a", 9.0, 0.90), ("b", 9.0, 0.90)],
        columns=["retriever", "nodes", "answer_node_recall"])
    _, recall, strategies = plots.identical_recall_cluster(frame)
    assert (recall, strategies) == (0.9, 4)


def test_no_cluster_is_reported_when_nothing_agrees():
    frame = pd.DataFrame([("a", 5.0, 0.10), ("b", 6.0, 0.42)],
                         columns=["retriever", "nodes", "answer_node_recall"])
    assert plots.identical_recall_cluster(frame) is None
    assert plots.identical_recall_cluster(frame.iloc[:0]) is None


# --- identity is never carried by colour alone ------------------------------

def test_each_strategy_has_a_distinct_colour_line_style_and_marker():
    styles = list(plots.SERIES_STYLE.values())
    for key in ("color", "linestyle", "marker"):
        values = [style[key] for style in styles]
        assert len(set(values)) == len(values), f"{key} repeats across strategies"


def test_a_strategy_keeps_one_appearance_however_the_run_named_it():
    # Result files label the same strategy differently across experiments.
    assert plots.style_for("PCST") == plots.style_for("PCST(topk=10, cost_e=0.5)")
    assert plots.style_for("top-k triples (KAPING)") == plots.style_for("Top-k triples")
    assert plots.style_for("something unmapped") is plots.FALLBACK_STYLE


def test_the_yellow_slot_is_not_used():
    # It is illegible as a thin line on white paper, which is the medium here.
    colours = {style["color"].lower() for style in plots.SERIES_STYLE.values()}
    assert "#eda100" not in colours


# --- the figures themselves -------------------------------------------------

def test_size_is_plotted_logarithmically(sweep):
    ax = plots.recall_vs_size(sweep).axes[0]
    assert ax.get_xscale() == "log", "sizes span three orders of magnitude"
    assert "node" in ax.get_xlabel().lower()
    assert "recall" in ax.get_ylabel().lower()


def test_every_strategy_appears_once_in_the_legend(sweep):
    ax = plots.recall_vs_size(sweep).axes[0]
    labels = [text.get_text() for text in ax.get_legend().get_texts()]
    assert sorted(labels) == sorted(sweep["retriever"].unique())


def test_recall_axis_covers_the_full_range_so_curves_are_comparable(sweep):
    bottom, top = plots.recall_vs_size(sweep).axes[0].get_ylim()
    assert bottom <= 0.0 and top >= 1.0


def test_a_measured_zero_is_labelled_because_it_draws_no_bar():
    gain = pd.DataFrame([
        ("catalogue (de)", "dx-codeonly", 0.80),
        ("catalogue (en), question de", "dx-codeonly", 0.00),
        ("catalogue (de)", "dx-named", 1.00),
        ("catalogue (en), question de", "dx-named", 0.00),
    ], columns=["condition", "qkind", "answer_node_recall"])

    ax = plots.terminology_gain(gain).axes[0]
    printed = {text.get_text() for text in ax.texts}
    assert "0.00" in printed, "a zero bar is invisible; the label is the only evidence"
    assert {"0.80", "1.00"} <= printed


def test_candidate_generation_shows_recall_and_ceiling_together():
    frame = pd.DataFrame([
        ("whole graph", 0.164, 1.000),
        ("seed expansion, cap 500", 0.164, 0.643),
    ], columns=["condition", "recall", "ceiling"])

    ax = plots.recall_and_ceiling(frame).axes[0]
    labels = [text.get_text().lower() for text in ax.get_legend().get_texts()]
    assert any("ceiling" in label for label in labels)
    assert any("recall" in label for label in labels)


def test_long_condition_names_go_on_the_vertical_axis():
    # Horizontal bars: these labels are phrases and collide when rotated.
    frame = pd.DataFrame([
        ("a condition with a very long descriptive name", 0.5, 0.9),
        ("another condition with an equally long name", 0.4, 0.8),
    ], columns=["condition", "recall", "ceiling"])

    ax = plots.recall_and_ceiling(frame).axes[0]
    tick_labels = [text.get_text() for text in ax.get_yticklabels()]
    assert any("very long descriptive name" in label for label in tick_labels)


def test_figures_are_written_as_vector_pdf(tmp_path, sweep):
    written = plots.save(plots.recall_vs_size(sweep), tmp_path / "curve")
    suffixes = {path.suffix for path in written}
    assert ".pdf" in suffixes, "LaTeX embeds PDF without resampling"
    for path in written:
        assert path.exists() and path.stat().st_size > 0
