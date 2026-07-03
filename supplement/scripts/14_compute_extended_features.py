"""Compute extended features for AUC optimization (Phase A+B).

Reads the Phase 1 dataset (sorted by add_date) and adds ~22 new features:

A. Developer-project co-occurrence (3, temporal):
   - dev_proj_past_satd_count
   - dev_proj_past_sf_rate
   - dev_proj_past_sf_count

B. Project-level temporal target encoding (2):
   - proj_hist_sf_rate
   - proj_hist_satd_count

C. Comment content features (5, from f_comment):
   - comment_has_fixme, comment_has_issue_ref, comment_has_url
   - comment_has_question, comment_keyword_type

D. Code context features (5, from f_path/method_declaration):
   - is_test_file, is_test_method, method_visibility
   - is_void_method, is_static_method

E. Derived / interaction features (7):
   - developer_satd_density, developer_fix_speed
   - developer_is_top_committer
   - ownership_x_active_commits, ownership_x_contribution_ratio
   - self_fix_rate_x_ownership
   - log1p_developer_total_commits

All temporal features use only information available before the record's add_date.
"""

from __future__ import annotations

import argparse
import json
import re
from bisect import bisect_right
from collections import defaultdict
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PLAN_DIR = SCRIPT_DIR.parent
DEFAULT_INPUT = PLAN_DIR / "data" / "merged_data_40501_phase1.json"
DEFAULT_OUTPUT = PLAN_DIR / "data" / "merged_data_40501_extended.json"


def load_data(path: Path) -> list[dict]:
    """Load JSON dataset."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def compute_temporal_features(records: list[dict]) -> None:
    """Add developer-project co-occurrence and project-level target encoding features."""
    dev_proj_resolved: dict[tuple, list[dict]] = defaultdict(list)
    proj_resolved: dict[str, list[dict]] = defaultdict(list)

    for r in records:
        email = (r.get("adder_email") or "").strip().lower()
        proj = r["project_name"]
        if r.get("remove_date"):
            entry = {"remove_date": r["remove_date"], "sf": r.get("is_self_fixed", 0)}
            if email:
                dev_proj_resolved[(email, proj)].append(entry)
            proj_resolved[proj].append(entry)

    for key in dev_proj_resolved:
        dev_proj_resolved[key].sort(key=lambda x: x["remove_date"])
    for proj in proj_resolved:
        proj_resolved[proj].sort(key=lambda x: x["remove_date"])

    global_sf_rate = sum(r.get("is_self_fixed", 0) for r in records) / len(records)

    for r in records:
        email = (r.get("adder_email") or "").strip().lower()
        proj = r["project_name"]
        add_date = r.get("add_date", "")

        dp_key = (email, proj)
        if email and dp_key in dev_proj_resolved:
            history = dev_proj_resolved[dp_key]
            rds = [h["remove_date"] for h in history]
            cutoff = bisect_right(rds, add_date)
            past = history[:cutoff]
            dp_total = len(past)
            dp_sf = sum(h["sf"] for h in past)
        else:
            dp_total = 0
            dp_sf = 0

        r["dev_proj_past_satd_count"] = dp_total
        r["dev_proj_past_sf_rate"] = dp_sf / dp_total if dp_total > 0 else global_sf_rate
        r["dev_proj_past_sf_count"] = dp_sf

        if proj in proj_resolved:
            p_history = proj_resolved[proj]
            p_rds = [h["remove_date"] for h in p_history]
            p_cutoff = bisect_right(p_rds, add_date)
            p_past = p_history[:p_cutoff]
            p_total = len(p_past)
            p_sf = sum(h["sf"] for h in p_past)
        else:
            p_total = 0
            p_sf = 0

        r["proj_hist_sf_rate"] = p_sf / p_total if p_total > 0 else global_sf_rate
        r["proj_hist_satd_count"] = p_total


RE_FIXME = re.compile(r"\bFIXME\b", re.IGNORECASE)
RE_TODO = re.compile(r"\bTODO\b", re.IGNORECASE)
RE_HACK = re.compile(r"\bHACK\b", re.IGNORECASE)
RE_XXX = re.compile(r"\bXXX\b", re.IGNORECASE)
RE_ISSUE = re.compile(r"(#\d+|[A-Z]{2,}-\d+)")
RE_URL = re.compile(r"https?://")
RE_QUESTION = re.compile(r"\?")


def compute_comment_features(records: list[dict]) -> None:
    """Extract keyword-based features from f_comment."""
    for r in records:
        comment = r.get("f_comment") or ""
        r["comment_has_fixme"] = 1 if RE_FIXME.search(comment) else 0
        r["comment_has_issue_ref"] = 1 if RE_ISSUE.search(comment) else 0
        r["comment_has_url"] = 1 if RE_URL.search(comment) else 0
        r["comment_has_question"] = 1 if RE_QUESTION.search(comment) else 0

        if RE_FIXME.search(comment):
            r["comment_keyword_type"] = "FIXME"
        elif RE_HACK.search(comment):
            r["comment_keyword_type"] = "HACK"
        elif RE_XXX.search(comment):
            r["comment_keyword_type"] = "XXX"
        elif RE_TODO.search(comment):
            r["comment_keyword_type"] = "TODO"
        else:
            r["comment_keyword_type"] = "OTHER"


def compute_code_context_features(records: list[dict]) -> None:
    """Extract features from f_path, method_declaration, containing_method."""
    for r in records:
        fpath = (r.get("f_path") or "").lower()
        r["is_test_file"] = 1 if "/test/" in fpath or "/tests/" in fpath else 0

        method_name = (r.get("containing_method") or "").lower()
        r["is_test_method"] = 1 if method_name.startswith("test") else 0

        decl = r.get("method_declaration") or ""
        if "private " in decl:
            r["method_visibility"] = "private"
        elif "protected " in decl:
            r["method_visibility"] = "protected"
        elif "public " in decl:
            r["method_visibility"] = "public"
        else:
            r["method_visibility"] = "default"

        r["is_void_method"] = 1 if "void " in decl else 0
        r["is_static_method"] = 1 if "static " in decl else 0


def compute_derived_features(records: list[dict]) -> None:
    """Compute interaction and derived features from existing fields."""
    proj_commits = defaultdict(list)
    for r in records:
        proj_commits[r["project_name"]].append(r.get("developer_total_commits", 0) or 0)

    proj_commit_threshold = {}
    for proj, commits in proj_commits.items():
        arr = np.array(commits)
        proj_commit_threshold[proj] = np.percentile(arr, 90) if len(arr) > 0 else 0

    for r in records:
        tc = r.get("developer_total_commits", 0) or 0
        ac = r.get("developer_active_commits", 0) or 0
        asc = r.get("developer_added_satd_count", 0) or 0
        lrd = r.get("developer_last_remove_satd_days", 0) or 0
        own = r.get("developer_ownership", 0) or 0
        cr = r.get("developer_contribution_ratio", 0) or 0
        sfr = r.get("developer_self_fix_rate", 0) or 0

        r["developer_satd_density"] = round(asc / (tc + 1), 6)
        r["developer_fix_speed"] = round(1.0 / (lrd + 1), 6)
        r["developer_is_top_committer"] = 1 if tc >= proj_commit_threshold.get(r["project_name"], 0) else 0

        r["ownership_x_active_commits"] = round(own * ac, 4)
        r["ownership_x_contribution_ratio"] = round(own * cr, 6)
        r["self_fix_rate_x_ownership"] = round(sfr * own, 6)

        r["log1p_developer_total_commits"] = round(float(np.log1p(tc)), 4)


def print_summary(records: list[dict], new_features: list[str]) -> None:
    """Print summary statistics for new features."""
    print(f"\nExtended dataset: {len(records)} records, {len(records[0])} fields")
    print(f"New features ({len(new_features)}):")
    for feat in new_features:
        vals = [r.get(feat) for r in records if r.get(feat) is not None]
        if not vals:
            print(f"  {feat}: all None")
            continue
        if isinstance(vals[0], (int, float)):
            arr = np.array(vals, dtype=float)
            print(f"  {feat}: mean={arr.mean():.4f}, median={np.median(arr):.4f}, "
                  f"min={arr.min():.4f}, max={arr.max():.4f}")
        else:
            from collections import Counter
            counts = Counter(vals)
            top3 = counts.most_common(3)
            print(f"  {feat}: {len(counts)} categories, top={top3}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    records = load_data(args.input)
    print(f"Loaded {len(records)} records from {args.input.name}")
    print(f"Date range: {records[0]['add_date']} -> {records[-1]['add_date']}")

    print("\nComputing temporal features (dev-proj co-occurrence + project encoding)...")
    compute_temporal_features(records)

    print("Computing comment content features...")
    compute_comment_features(records)

    print("Computing code context features...")
    compute_code_context_features(records)

    print("Computing derived/interaction features...")
    compute_derived_features(records)

    new_features = [
        "dev_proj_past_satd_count", "dev_proj_past_sf_rate", "dev_proj_past_sf_count",
        "proj_hist_sf_rate", "proj_hist_satd_count",
        "comment_has_fixme", "comment_has_issue_ref", "comment_has_url",
        "comment_has_question", "comment_keyword_type",
        "is_test_file", "is_test_method", "method_visibility",
        "is_void_method", "is_static_method",
        "developer_satd_density", "developer_fix_speed",
        "developer_is_top_committer",
        "ownership_x_active_commits", "ownership_x_contribution_ratio",
        "self_fix_rate_x_ownership",
        "log1p_developer_total_commits",
    ]
    print_summary(records, new_features)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"\nWrote extended dataset: {args.output}")
    print(f"Total fields per record: {len(records[0])}")


if __name__ == "__main__":
    main()
