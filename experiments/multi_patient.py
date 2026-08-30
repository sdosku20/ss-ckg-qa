"""
Does the evaluation hold up across patients, or only on the one we measured?

    python experiments/multi_patient.py                 # 8 patients, full size
    python experiments/multi_patient.py --patients 24   # more of them
    python experiments/multi_patient.py --small         # 10x smaller, for iteration

Every result in this project so far comes from a single patient graph. That
establishes mechanism and nothing more: patient #0 might be unusually large,
unusually rich in shared text, or unusually well described, and each of those
drives a reported number. This script runs the same retrievers over many
patients and reports the spread as well as the mean, because a method that works
on average and fails on a third of patients is not the same as a method that
works.

The patients here are replica draws. Each has its own random seed, and by
default each is also scaled to a different size, because two draws from one
profile differ only in content: their answer-set sizes are identical, so their
recall is identical to four decimal places and the cross-patient variance this
script exists to measure is exactly zero. Size is the axis a real cohort varies
on most.

The size distribution used here is assumed, not measured. What the 1,197 STCS
patients actually look like is item 2 of docs/server_measurements.md section 6
and needs a query to settle. So this answers "does the pipeline survive
patients of different sizes" and cannot answer "does it survive real clinical
variation". The loop is the part being tested, and it is the same loop the
cohort run will use with one argument changed.

Writes experiments/results/multi_patient{_small}.csv and can resume into it.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from ikgqa.data.replica import PATIENT_0, build_replica, planted_questions  # noqa: E402
from ikgqa.data.sphn import default_encoder  # noqa: E402
from ikgqa.eval.multipatient import PatientCase, run, summarise  # noqa: E402
from ikgqa.retrieval import PCST, TopKTriples  # noqa: E402
from ikgqa.retrieval.candidates import SeedExpansion, TwoStage  # noqa: E402

RESULTS = pathlib.Path(__file__).resolve().parent / "results"


#: Size multipliers cycled across patients. A placeholder for the real
#: distribution, wide enough that a method sensitive to graph size shows it.
#:
#: Capped at 1.2 because the replica's synthetic vocabulary runs out at 360
#: distinct drugs and patient #0 already uses 279. Real patients larger than the
#: measured one therefore cannot be simulated here, which is a limitation of the
#: fixture and one more reason the cohort size distribution has to be measured
#: on the server rather than assumed.
SIZE_FACTORS = (0.4, 0.6, 0.8, 1.0, 1.2)


def make_loader(profile, questions_per_kind: int, vary_size: bool = True):
    """Build one replica patient per identifier, deterministically.

    Seed and size factor both come from the patient id, so a rerun produces the
    same cohort and a resumed run agrees with the rows already on disk.
    """

    def load(patient: str):
        seed = int(patient.split("-")[-1])
        shape = profile.scaled(SIZE_FACTORS[seed % len(SIZE_FACTORS)]) if vary_size else profile
        graph, table, _stats, rows = build_replica(profile=shape, seed=seed)
        questions = planted_questions(graph, table, rows,
                                      n_per_kind=questions_per_kind, language="de")
        encoder = default_encoder(graph.embed_texts, graph.edges["edge_attr"].tolist())
        return PatientCase(graph=graph, questions=questions, encoder=encoder)

    return load


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate across many patients.")
    parser.add_argument("--patients", type=int, default=8)
    parser.add_argument("--questions", type=int, default=3,
                        help="planted questions per kind, per patient")
    parser.add_argument("--small", action="store_true", help="10x smaller patients")
    parser.add_argument("--same-size", action="store_true",
                        help="every patient the same shape; recall variance is then zero")
    parser.add_argument("--resume", action="store_true",
                        help="skip patients already present in the output file")
    args = parser.parse_args(argv)

    profile = PATIENT_0.scaled(0.1) if args.small else PATIENT_0
    suffix = "_small" if args.small else ""
    out = RESULTS / f"multi_patient{suffix}.csv"
    RESULTS.mkdir(parents=True, exist_ok=True)

    # Built per patient, not once: KAPING must be given that patient's encoder
    # or it ranks relation text alone, which understates the baseline.
    #
    # topk_e=0 throughout, matching experiments/sweep_replica.py. Leaving it at
    # its default of 5 re-enables edge prizes, and with few heavily shared
    # relation types those dilute until the per-edge cost is negligible and PCST
    # returns the entire graph. The first run of this script did exactly that
    # and scored a perfect 1.000 on every question of every patient, which is
    # what finding F2 looks like when it is not intended.
    def retrievers_for(case):
        inner = dict(topk=10, topk_e=0, cost_e=0.5, tie_break="stable")
        return [
            PCST(**inner),
            TwoStage(generator=SeedExpansion(k=10, hops=2, cap=2000),
                     inner=PCST(**inner)),
            TopKTriples(k=10, encoder=case.encoder),
        ]

    patients = [f"replica-{i}" for i in range(args.patients)]
    print(f"{len(patients)} patients, 3 retrievers, "
          f"{args.questions} questions per kind\n")

    started = time.time()
    frame, report = run(
        patients,
        make_loader(profile, args.questions, vary_size=not args.same_size),
        retrievers=retrievers_for,
        resume_from=str(out) if args.resume else None,
        progress=lambda line: print(f"  {line}"),
    )

    if args.resume and out.exists() and not frame.empty:
        frame = pd.concat([pd.read_csv(out), frame], ignore_index=True)
    if not frame.empty:
        frame.to_csv(out, index=False)

    print(f"\n{report.summary()}")
    if frame.empty:
        print("no rows produced")
        return 1

    print(f"\nwrote {out}  ({len(frame)} rows in {time.time() - started:.1f}s)\n")
    summary = summarise(frame)
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    spread = summary["recall_sd"].max()
    print(f"\nWidest spread across patients: {spread:.4f}")
    print("A large spread means the single-patient results in Chapter 6 describe "
          "that patient\nrather than the method.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
