"""
Build every figure in the thesis from the committed result files.

    python experiments/figures.py            # full-size results
    python experiments/figures.py --small    # the faster replica run

Reads only from experiments/results/ and writes only to thesis/Figures/, so a
figure can never disagree with the table beside it: both come from the same CSV.
Re-run this after any sweep and the chapter updates.

Nothing here touches the database or the replica generator. If a result file is
missing the figure is skipped with a message naming the script that produces it,
because a stale figure is worse than an absent one.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from ikgqa.eval import plots  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS = ROOT / "experiments" / "results"
FIGURES = ROOT / "thesis" / "Figures"

# figure stem -> (result file, builder, the script that regenerates the file)
FIGURE_SOURCES = {
    "recall-vs-size": ("replica_sweep{suffix}.csv", plots.recall_vs_size,
                       "experiments/sweep_replica.py"),
    "pcst-size-dial": ("replica_sweep{suffix}.csv", plots.size_reachable_by_pcst,
                       "experiments/sweep_replica.py"),
    "candidate-generation": ("candidate_generation{suffix}.csv", plots.recall_and_ceiling,
                             "experiments/candidate_generation.py"),
    "terminology-gain": ("terminology_gain{suffix}.csv", plots.terminology_gain,
                         "experiments/terminology_gain.py"),
}


def build(suffix: str = "_full") -> int:
    written, skipped = 0, 0

    for stem, (pattern, builder, source) in FIGURE_SOURCES.items():
        path = RESULTS / pattern.format(suffix=suffix)
        if not path.exists():
            print(f"  skip  {stem:22s} no {path.name}; run {source}")
            skipped += 1
            continue

        frame = pd.read_csv(path)
        figure = builder(frame)
        targets = plots.save(figure, FIGURES / stem)
        print(f"  wrote {stem:22s} {', '.join(t.name for t in targets)}"
              f"   ({len(frame)} rows from {path.name})")
        written += 1

        import matplotlib.pyplot as plt
        plt.close(figure)

    print(f"\n{written} figures written to {FIGURES}, {skipped} skipped")
    return 0 if written else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--small", action="store_true",
                        help="use the reduced replica results")
    args = parser.parse_args(argv)
    return build("_small" if args.small else "_full")


if __name__ == "__main__":
    sys.exit(main())
