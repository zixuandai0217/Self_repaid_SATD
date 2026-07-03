"""Compute 7 new features from merged_data JSON and produce an augmented sorted dataset.

New features:
  Developer dimension (4):
    - developer_self_fixed_count: number of self-fixed SATD by same developer (by email)
    - developer_self_fix_rate:  self_fixed_count / max(total_appearances, 1)
    - developer_fix_ratio:      removed_satd_count / max(added_satd_count, 1)
    - developer_contribution_ratio: total_commits / max(project_total_commits, 1)
  Comment/code dimension (3):
    - f_comment_type:   already present in JSON, just needs to be used (Block/Line)
    - comment_span:     end_line - start_line + 1
    - method_body_lines: line count of method_body field
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


PLAN_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PLAN_DIR / "data" / "merged_data_40501_temporal_real.json"
DEFAULT_OUTPUT = PLAN_DIR / "data" / "merged_data_40501_phase1.json"
DEFAULT_SORTED_OUTPUT = PLAN_DIR / "data" / "merged_data_40501_temporal_sorted_by_add_date.json"


def load_data(path: Path) -> list[dict]:
    """Load the JSON dataset."""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} records from {path.name}")
    return data


def safe_divide(numerator, denominator, default=0.0):
    """Division with zero-denominator protection."""
    return numerator / denominator if denominator and denominator > 0 else default


def augment_records(records: list[dict]) -> list[dict]:
    """Add 7 new features to each record and return sorted by add_date.

    developer_self_fixed_count and developer_self_fix_rate are computed
    TEMPORALLY: for each record, only count self-fixed SATD whose remove_date
    is strictly before the current record's add_date, preventing data leakage.
    """
    records.sort(key=lambda x: x.get("add_date", ""))

    dev_resolved: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        email = (r.get("adder_email") or "").strip().lower()
        if email and r.get("remove_date"):
            dev_resolved[email].append({
                "remove_date": r["remove_date"],
                "is_self_fixed": r.get("is_self_fixed", 0),
            })

    for email in dev_resolved:
        dev_resolved[email].sort(key=lambda x: x["remove_date"])

    from bisect import bisect_right

    for r in records:
        email = (r.get("adder_email") or "").strip().lower()
        add_date = r.get("add_date", "")

        if email and email in dev_resolved:
            history = dev_resolved[email]
            remove_dates = [h["remove_date"] for h in history]
            cutoff = bisect_right(remove_dates, add_date)
            past_records = history[:cutoff]
            past_total = len(past_records)
            past_self_fixed = sum(1 for h in past_records if h["is_self_fixed"] == 1)
        else:
            past_total = 0
            past_self_fixed = 0

        r["developer_self_fixed_count"] = past_self_fixed
        r["developer_self_fix_rate"] = round(safe_divide(past_self_fixed, max(past_total, 1)), 6)

        added = r.get("developer_added_satd_count") or 0
        removed = r.get("developer_removed_satd_count") or 0
        r["developer_fix_ratio"] = round(safe_divide(removed, max(added, 1)), 6)

        dev_commits = r.get("developer_total_commits") or 0
        proj_commits = r.get("project_total_commits") or 0
        r["developer_contribution_ratio"] = round(safe_divide(dev_commits, max(proj_commits, 1)), 6)

        start = r.get("start_line") or 0
        end = r.get("end_line") or 0
        r["comment_span"] = max(end - start + 1, 1)

        body = r.get("method_body") or ""
        r["method_body_lines"] = body.count("\n") + 1 if body.strip() else 0

    return records


def print_summary(records: list[dict]) -> None:
    """Print distribution summary for new features."""
    import statistics

    features = [
        "developer_self_fixed_count",
        "developer_self_fix_rate",
        "developer_fix_ratio",
        "developer_contribution_ratio",
        "comment_span",
        "method_body_lines",
    ]
    for feat in features:
        vals = [r[feat] for r in records if r.get(feat) is not None]
        if not vals:
            print(f"  {feat}: NO DATA")
            continue
        print(
            f"  {feat}: min={min(vals):.3f}, max={max(vals):.3f}, "
            f"median={statistics.median(vals):.3f}, mean={statistics.mean(vals):.3f}"
        )

    from collections import Counter
    ct = Counter(r.get("f_comment_type") for r in records)
    print(f"  f_comment_type: {dict(ct)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sorted-output", type=Path, default=DEFAULT_SORTED_OUTPUT)
    args = parser.parse_args()

    records = load_data(args.input)
    records = augment_records(records)

    print(f"\nNew feature summary ({len(records)} records):")
    print_summary(records)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"\nWrote augmented dataset: {args.output}")

    with args.sorted_output.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"Wrote sorted dataset: {args.sorted_output}")
    print(f"Date range: {records[0]['add_date']} -> {records[-1]['add_date']}")


if __name__ == "__main__":
    main()
