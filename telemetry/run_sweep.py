"""Phase 1 pipeline: sweep NFE values, time each, save a run-record per NFE."""

import argparse
import csv
import os
from datetime import datetime
from functools import partial

from .record import EST, OUTPUTS_DIR, RunRecord
from .timing import time_callable
from .toy_model import run_toy_model

DEFAULT_NFES = [4, 8, 16, 32, 64]
DEFAULT_WARMUPS = 3
DEFAULT_REPEATS = 10
SUMMARY_FIELDS = ["nfe", "median_ms", "per_step_ms", "min_ms", "max_ms", "run_id"]


def run_sweep(nfes, warmups=DEFAULT_WARMUPS, repeats=DEFAULT_REPEATS, out_dir=OUTPUTS_DIR):
    """Time the toy model at each NFE; save one run-record each plus a CSV summary.

    Returns (rows, summary_path).
    """
    sweep_id = datetime.now(EST).strftime("%Y%m%d_%H%M%S")
    print(f"sweep {sweep_id}: nfe={nfes} warmups={warmups} repeats={repeats}\n")
    print(f"{'NFE':>5}  {'median ms':>10}  {'per-step ms':>11}")
    print(f"{'-' * 5}  {'-' * 10}  {'-' * 11}")

    rows = []
    for nfe in nfes:
        record = RunRecord(config={"model": "toy", "nfe": nfe, "sweep_id": sweep_id})
        timing = time_callable(partial(run_toy_model, nfe), nfe, warmups, repeats)
        record.metrics["timing"] = timing
        record.save(out_dir)

        rows.append({
            "nfe": nfe,
            "median_ms": timing["median_ms"],
            "per_step_ms": timing["per_step_ms"],
            "min_ms": timing["min_ms"],
            "max_ms": timing["max_ms"],
            "run_id": record.run_id,
        })
        print(f"{nfe:>5}  {timing['median_ms']:>10.2f}  {timing['per_step_ms']:>11.3f}")

    summary_path = os.path.join(out_dir, f"sweep_{sweep_id}.csv")
    # newline="" stops Windows turning csv's \r\n into \r\r\n (blank rows).
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return rows, summary_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nfe", type=int, nargs="+", default=DEFAULT_NFES,
                        help=f"NFE values to sweep (default: {DEFAULT_NFES})")
    parser.add_argument("--warmups", type=int, default=DEFAULT_WARMUPS)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    args = parser.parse_args()
    if min(args.nfe) < 1 or args.warmups < 0 or args.repeats < 1:
        parser.error("need every --nfe >= 1, --warmups >= 0, --repeats >= 1")

    _, summary_path = run_sweep(args.nfe, args.warmups, args.repeats)
    print(f"\nsummary: {summary_path}")


if __name__ == "__main__":
    main()
