from __future__ import annotations

import argparse
import csv
from pathlib import Path

from temporal_utils import summarize_manual_validation


PLAN_DIR = Path(__file__).resolve().parents[1]
SELF_REPAID_ROOT = next(parent for parent in PLAN_DIR.parents if parent.name == "Self Repaid SATD")
DEFAULT_SAMPLE_CSV = SELF_REPAID_ROOT / "code" / "modeling" / "analysis" / "366_human_eval_sample_384.csv"
DEFAULT_OUTPUT_MD = PLAN_DIR / "results" / "manual_satd_validation.md"
DEFAULT_POPULATION_SIZE = 40501


def read_validation_rows(path: Path) -> list[dict]:
    """Read the manually labeled SATD validation sample."""

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if "manual_is_satd" not in (reader.fieldnames or []):
        raise ValueError(f"`manual_is_satd` column not found in {path}")
    return rows


def write_manual_validation_report(sample_path: Path, output_path: Path, population_size: int) -> None:
    """Write precision and Wilson CI for the manual SATD validation sample."""

    rows = read_validation_rows(sample_path)
    summary = summarize_manual_validation(rows, population_size=population_size)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Manual SATD Validation",
        "",
        "## Input",
        f"- Manual validation CSV: `{sample_path}`",
        f"- Population size: {summary.population_size:,}",
        f"- Sample size: {summary.sample_size:,}",
        "",
        "## Sampling Rationale",
        "",
        "The validation sample size follows the finite-population formula for estimating a population proportion with 95% confidence, a 5% margin of error, and the conservative setting p = 0.5. For N = 40,501, the required sample size is approximately "
        f"{summary.required_sample_size:.2f}; the 384-record sample therefore satisfies this design.",
        "",
        "## Result",
        "",
        f"- Manually confirmed SATD records: {summary.confirmed_satd:,}/{summary.sample_size:,}",
        f"- Precision: {summary.precision * 100:.1f}%",
        f"- Wilson 95% CI: {summary.wilson_low * 100:.1f}% to {summary.wilson_high * 100:.1f}%",
        "",
        "## Paper-Ready Wording",
        "",
        "To assess the data quality of the final SATD instances, we randomly sampled 384 records from the 40,501 instances. The sample size was determined using a 95% confidence level, a 5% margin of error, and a conservative population proportion of p = 0.5. Manual inspection confirmed that 366 of the 384 sampled records were true SATD instances, yielding a precision of 95.3% with a Wilson 95% confidence interval of 92.7% to 97.0%. This result indicates that the final SATD dataset has high precision for the purposes of the empirical analysis.",
        "",
        "## Suggested Placement",
        "",
        "- Data Quality subsection: report the sampling rationale and precision.",
        "- Threats to Validity subsection: state that manual validation reduces concern about SATD false positives in the final dataset, while acknowledging that recall is not estimated by this precision-oriented sample.",
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Validate the existing 384-record manual SATD sample."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-csv", type=Path, default=DEFAULT_SAMPLE_CSV)
    parser.add_argument("--population-size", type=int, default=DEFAULT_POPULATION_SIZE)
    parser.add_argument("--report", type=Path, default=DEFAULT_OUTPUT_MD)
    args = parser.parse_args()

    write_manual_validation_report(args.sample_csv, args.report, args.population_size)
    print(f"Wrote manual validation report: {args.report}")


if __name__ == "__main__":
    main()
