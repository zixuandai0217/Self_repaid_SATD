from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Sequence


DEFAULT_TRAIN_PERCENTAGES = tuple(range(10, 100, 10))
DEFAULT_FIXED_TEST_TRAIN_PERCENTAGES = tuple(range(10, 90, 10))


@dataclass(frozen=True)
class TemporalSplit:
    """Describe one chronological train/test split."""

    train_percentage: int
    train_size: int
    test_size: int
    train_start_index: int
    train_end_index: int
    test_start_index: int
    test_end_index: int
    validation_size: int = 0
    validation_start_index: int = -1
    validation_end_index: int = -1
    train_start_date: str | None = None
    train_end_date: str | None = None
    validation_start_date: str | None = None
    validation_end_date: str | None = None
    test_start_date: str | None = None
    test_end_date: str | None = None
    train_positive: int = 0
    train_negative: int = 0
    validation_positive: int = 0
    validation_negative: int = 0
    test_positive: int = 0
    test_negative: int = 0


@dataclass(frozen=True)
class ManualValidationSummary:
    """Summarize manual SATD validation for sampled records."""

    population_size: int
    sample_size: int
    confirmed_satd: int
    precision: float
    wilson_low: float
    wilson_high: float
    required_sample_size: float


def parse_add_date(value: object) -> datetime:
    """Parse add_date values used by the SATD dataset."""

    if value is None:
        raise ValueError("add_date is missing")
    text = str(value).strip()
    if not text:
        raise ValueError("add_date is empty")

    candidates = [
        text,
        text.replace("/", "-"),
    ]
    formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d",
    ]
    for candidate in candidates:
        for fmt in formats:
            try:
                return datetime.strptime(candidate, fmt)
            except ValueError:
                pass
    try:
        return datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"Unsupported add_date format: {text!r}") from exc


def stable_sort_records_by_add_date(records: Sequence[dict]) -> list[dict]:
    """Return records sorted by add_date while preserving original tie order."""

    indexed = list(enumerate(records))
    indexed.sort(key=lambda item: (parse_add_date(item[1].get("add_date")), item[0]))
    return [record for _, record in indexed]


def field_set(records: Sequence[dict]) -> set[str]:
    """Return the union of fields present in the supplied records."""

    fields: set[str] = set()
    for record in records:
        fields.update(record.keys())
    return fields


def is_monotonic_by_add_date(records: Sequence[dict]) -> bool:
    """Check whether records are non-decreasing by add_date."""

    previous: datetime | None = None
    for record in records:
        current = parse_add_date(record.get("add_date"))
        if previous is not None and current < previous:
            return False
        previous = current
    return True


def _count_labels(records: Sequence[dict]) -> tuple[int, int]:
    positives = sum(1 for record in records if int(record.get("is_self_fixed", 0)) == 1)
    negatives = len(records) - positives
    return positives, negatives


def _boundary_key(record: dict) -> tuple[object, object]:
    """Return the temporal grouping key that should not be split across windows."""

    add_date = record.get("add_date")
    add_commit_hash = record.get("add_commit_hash")
    if add_date in {None, ""} and add_commit_hash in {None, ""}:
        return "__missing_temporal_key__", record.get("satd_id")
    return add_date, add_commit_hash


def _advance_boundary_past_tie(records: Sequence[dict], boundary: int) -> int:
    """Move a split boundary forward so one add-date/commit group stays in one window."""

    if boundary <= 0 or boundary >= len(records):
        return boundary
    previous_key = _boundary_key(records[boundary - 1])
    while boundary < len(records) and _boundary_key(records[boundary]) == previous_key:
        boundary += 1
    return boundary


def make_temporal_splits(
    records: Sequence[dict],
    train_percentages: Iterable[int] = DEFAULT_TRAIN_PERCENTAGES,
) -> list[TemporalSplit]:
    """Create global chronological expanding-window splits."""

    n_records = len(records)
    if n_records < 2:
        raise ValueError("At least two records are required for temporal splitting")

    splits: list[TemporalSplit] = []
    for percentage in train_percentages:
        if percentage <= 0 or percentage >= 100:
            raise ValueError(f"Train percentage must be between 1 and 99: {percentage}")
        train_size = int(n_records * percentage / 100)
        if train_size <= 0 or train_size >= n_records:
            raise ValueError(
                f"Train percentage {percentage} creates an invalid split for {n_records} records"
            )
        test_size = n_records - train_size
        train_records = records[:train_size]
        test_records = records[train_size:]
        train_positive, train_negative = _count_labels(train_records)
        test_positive, test_negative = _count_labels(test_records)
        splits.append(
            TemporalSplit(
                train_percentage=percentage,
                train_size=train_size,
                test_size=test_size,
                train_start_index=0,
                train_end_index=train_size - 1,
                test_start_index=train_size,
                test_end_index=n_records - 1,
                train_start_date=str(train_records[0].get("add_date")),
                train_end_date=str(train_records[-1].get("add_date")),
                test_start_date=str(test_records[0].get("add_date")),
                test_end_date=str(test_records[-1].get("add_date")),
                train_positive=train_positive,
                train_negative=train_negative,
                test_positive=test_positive,
                test_negative=test_negative,
            )
        )
    return splits


def make_fixed_test_temporal_splits(
    records: Sequence[dict],
    train_percentages: Iterable[int] = DEFAULT_FIXED_TEST_TRAIN_PERCENTAGES,
    validation_start_percentage: int = 80,
    test_start_percentage: int = 90,
) -> list[TemporalSplit]:
    """Create chronological splits with fixed 80%-90% validation and 90%-100% test windows."""

    n_records = len(records)
    if n_records < 10:
        raise ValueError("At least ten records are required for 8:1:1 temporal splitting")
    if not (0 < validation_start_percentage < test_start_percentage < 100):
        raise ValueError("Expected 0 < validation_start_percentage < test_start_percentage < 100")

    validation_start = _advance_boundary_past_tie(
        records, int(n_records * validation_start_percentage / 100)
    )
    test_start = _advance_boundary_past_tie(records, int(n_records * test_start_percentage / 100))
    if validation_start <= 0 or validation_start >= n_records:
        raise ValueError("validation_start_percentage creates an invalid validation window")
    if test_start <= validation_start or test_start >= n_records:
        raise ValueError("test_start_percentage creates an invalid test window")

    validation_records = records[validation_start:test_start]
    test_records = records[test_start:]
    validation_positive, validation_negative = _count_labels(validation_records)
    test_positive, test_negative = _count_labels(test_records)

    splits: list[TemporalSplit] = []
    for percentage in train_percentages:
        if percentage <= 0 or percentage > validation_start_percentage:
            raise ValueError(
                "Train percentage must be between 1 and the validation start percentage "
                f"({validation_start_percentage}): {percentage}"
            )
        train_size = _advance_boundary_past_tie(records, int(n_records * percentage / 100))
        if train_size <= 0 or train_size > validation_start:
            raise ValueError(
                f"Train percentage {percentage} creates an invalid split for {n_records} records"
            )

        train_records = records[:train_size]
        train_positive, train_negative = _count_labels(train_records)
        splits.append(
            TemporalSplit(
                train_percentage=percentage,
                train_size=train_size,
                validation_size=len(validation_records),
                test_size=len(test_records),
                train_start_index=0,
                train_end_index=train_size - 1,
                validation_start_index=validation_start,
                validation_end_index=test_start - 1,
                test_start_index=test_start,
                test_end_index=n_records - 1,
                train_start_date=str(train_records[0].get("add_date")),
                train_end_date=str(train_records[-1].get("add_date")),
                validation_start_date=str(validation_records[0].get("add_date")),
                validation_end_date=str(validation_records[-1].get("add_date")),
                test_start_date=str(test_records[0].get("add_date")),
                test_end_date=str(test_records[-1].get("add_date")),
                train_positive=train_positive,
                train_negative=train_negative,
                validation_positive=validation_positive,
                validation_negative=validation_negative,
                test_positive=test_positive,
                test_negative=test_negative,
            )
        )
    return splits


def wilson_ci(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Compute a Wilson score confidence interval for a binomial proportion."""

    if total <= 0:
        raise ValueError("total must be positive")
    if successes < 0 or successes > total:
        raise ValueError("successes must be between 0 and total")
    p_hat = successes / total
    denominator = 1 + z**2 / total
    center = (p_hat + z**2 / (2 * total)) / denominator
    half_width = (
        z
        * math.sqrt((p_hat * (1 - p_hat) / total) + (z**2 / (4 * total**2)))
        / denominator
    )
    return center - half_width, center + half_width


def calculate_sample_size(
    population_size: int,
    confidence_z: float = 1.96,
    margin_error: float = 0.05,
    proportion: float = 0.5,
) -> float:
    """Calculate finite-population sample size for proportion estimation."""

    if population_size <= 0:
        raise ValueError("population_size must be positive")
    n0 = (confidence_z**2 * proportion * (1 - proportion)) / (margin_error**2)
    return (population_size * n0) / (population_size + n0 - 1)


def summarize_manual_validation(
    rows: Sequence[dict],
    population_size: int,
) -> ManualValidationSummary:
    """Summarize manual_is_satd labels from a sampled validation CSV."""

    sample_size = len(rows)
    if sample_size == 0:
        raise ValueError("manual validation rows cannot be empty")
    confirmed = 0
    for row in rows:
        value = str(row.get("manual_is_satd", "")).strip()
        if value in {"1", "1.0", "true", "True", "TRUE", "yes", "Yes"}:
            confirmed += 1
    precision = confirmed / sample_size
    low, high = wilson_ci(confirmed, sample_size)
    return ManualValidationSummary(
        population_size=population_size,
        sample_size=sample_size,
        confirmed_satd=confirmed,
        precision=precision,
        wilson_low=low,
        wilson_high=high,
        required_sample_size=calculate_sample_size(population_size),
    )
