from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from temporal_utils import (
    field_set,
    is_monotonic_by_add_date,
    make_temporal_splits,
    stable_sort_records_by_add_date,
)


PLAN_DIR = Path(__file__).resolve().parents[1]
SELF_REPAID_ROOT = next(parent for parent in PLAN_DIR.parents if parent.name == "Self Repaid SATD")
DEFAULT_SOURCE_JSON = (
    SELF_REPAID_ROOT
    / "code"
    / "data_preparation"
    / "data"
    / "merged_data_40501_updated_with_comment.json"
)
DEFAULT_OUTPUT_JSON = PLAN_DIR / "data" / "merged_data_40501_temporal_sorted_by_add_date.json"
DEFAULT_OUTPUT_MD = PLAN_DIR / "results" / "temporal_dataset_validation.md"


def load_json_records(path: Path) -> list[dict]:
    """Load the SATD JSON dataset as a list of records."""

    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list of records in {path}")
    if not all(isinstance(record, dict) for record in data):
        raise ValueError(f"Expected every record in {path} to be a JSON object")
    return data


def record_signature(record: dict) -> str:
    """Create a stable hash for one JSON record."""

    payload = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def describe_edge_records(records: list[dict], limit: int = 5) -> tuple[list[dict], list[dict]]:
    """Return compact first and last records for sorting evidence."""

    def compact(record: dict) -> dict:
        return {
            "satd_id": record.get("satd_id"),
            "project_name": record.get("project_name"),
            "add_date": record.get("add_date"),
            "is_self_fixed": record.get("is_self_fixed"),
        }

    return [compact(record) for record in records[:limit]], [compact(record) for record in records[-limit:]]


def write_sorted_dataset(records: list[dict], output_path: Path) -> None:
    """Write the temporal dataset copy without mutating the original source."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=False, separators=(",", ":"))
        handle.write("\n")


def write_validation_report(
    original_records: list[dict],
    sorted_records: list[dict],
    source_path: Path,
    output_path: Path,
    report_path: Path,
) -> None:
    """Write a Markdown report proving that only record order changed."""

    original_fields = field_set(original_records)
    sorted_fields = field_set(sorted_records)
    original_satd_ids = Counter(str(record.get("satd_id")) for record in original_records)
    sorted_satd_ids = Counter(str(record.get("satd_id")) for record in sorted_records)
    original_signatures = Counter(record_signature(record) for record in original_records)
    sorted_signatures = Counter(record_signature(record) for record in sorted_records)
    first_records, last_records = describe_edge_records(sorted_records)
    splits = make_temporal_splits(sorted_records)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Temporal Dataset Validation",
        "",
        "## Files",
        f"- Source JSON: `{source_path}`",
        f"- Temporal JSON: `{output_path}`",
        "",
        "## Integrity Checks",
        f"- Source records: {len(original_records):,}",
        f"- Temporal records: {len(sorted_records):,}",
        f"- Field set identical: {original_fields == sorted_fields}",
        f"- Number of fields: {len(original_fields)}",
        f"- `satd_id` multiset identical: {original_satd_ids == sorted_satd_ids}",
        f"- Full-record value multiset identical: {original_signatures == sorted_signatures}",
        f"- `add_date` monotonic non-decreasing: {is_monotonic_by_add_date(sorted_records)}",
        "",
        "## First 5 Records After Sorting",
        "",
        "| # | satd_id | project_name | add_date | is_self_fixed |",
        "|---|---:|---|---|---:|",
    ]
    for index, record in enumerate(first_records, 1):
        lines.append(
            f"| {index} | {record['satd_id']} | {record['project_name']} | "
            f"{record['add_date']} | {record['is_self_fixed']} |"
        )
    lines.extend(
        [
            "",
            "## Last 5 Records After Sorting",
            "",
            "| # | satd_id | project_name | add_date | is_self_fixed |",
            "|---|---:|---|---|---:|",
        ]
    )
    for index, record in enumerate(last_records, 1):
        lines.append(
            f"| {index} | {record['satd_id']} | {record['project_name']} | "
            f"{record['add_date']} | {record['is_self_fixed']} |"
        )
    lines.extend(
        [
            "",
            "## Temporal Split Plan",
            "",
            "| Train % | Train n | Test n | Train date range | Test date range | Train positive/negative | Test positive/negative |",
            "|---:|---:|---:|---|---|---:|---:|",
        ]
    )
    for split in splits:
        lines.append(
            f"| {split.train_percentage}% | {split.train_size:,} | {split.test_size:,} | "
            f"{split.train_start_date} to {split.train_end_date} | "
            f"{split.test_start_date} to {split.test_end_date} | "
            f"{split.train_positive}/{split.train_negative} | "
            f"{split.test_positive}/{split.test_negative} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This temporal copy changes only the order of records. Records are sorted by global `add_date` in ascending order, and ties are resolved by the original row order. The sorted dataset is therefore suitable for expanding-window evaluation while preserving the original fields and values.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Create the temporal SATD dataset copy and validation report."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-json", type=Path, default=DEFAULT_SOURCE_JSON)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--report", type=Path, default=DEFAULT_OUTPUT_MD)
    args = parser.parse_args()

    original_records = load_json_records(args.source_json)
    sorted_records = stable_sort_records_by_add_date(original_records)

    write_sorted_dataset(sorted_records, args.output_json)
    write_validation_report(
        original_records=original_records,
        sorted_records=sorted_records,
        source_path=args.source_json,
        output_path=args.output_json,
        report_path=args.report,
    )
    print(f"Wrote temporal dataset: {args.output_json}")
    print(f"Wrote validation report: {args.report}")


if __name__ == "__main__":
    main()
