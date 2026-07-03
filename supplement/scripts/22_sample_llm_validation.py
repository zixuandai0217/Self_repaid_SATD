"""
Sample instances for quality_score and past_*_ratio human validation.

Phase 1 (quality_score): Extracts from existing 384-sample CSV (366 confirmed SATD).
Phase 2 (past_*_ratio): Randomly selects SATD instances from the dataset, retrieves
    the identical prior-commit sequences via GitHub Commits API (same author + date
    parameters as the original git-log pipeline), classifies with DeepSeek-V3,
    and outputs annotation CSV.

Reproducibility:
    - All random operations use RANDOM_SEED = 2025.
    - Phase 2 writes a provenance manifest (validation_commit_provenance.json) that
      records every SATD instance sampled, the API parameters used, and the commits
      returned, so that results can be independently verified.
    - Intermediate classified commits are cached in validation_commit_messages_raw.json;
      re-running the script reuses the cache and skips both API and LLM calls.

Usage:
    python 22_sample_llm_validation.py --phase quality    # Phase 1 only
    python 22_sample_llm_validation.py --phase commit     # Phase 2 only
    python 22_sample_llm_validation.py --phase all        # Both phases
"""

import json
import csv
import os
import sys
import random
import argparse
import time
import re
import math
import urllib.request
import urllib.error
from collections import Counter
from datetime import datetime

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SELF_REPAID_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "../../../../.."))

SAMPLE_384_CSV = os.path.join(
    SELF_REPAID_ROOT, "code/modeling/analysis/366_human_eval_sample_384.csv"
)
RAW_DATA_JSON = os.path.join(
    SELF_REPAID_ROOT, "code/data_preparation/data/raw_data_final_40501.json"
)
MERGED_DATA_JSON = os.path.join(
    SELF_REPAID_ROOT, "code/data_preparation/data/merged_data_40501.json"
)

OUTPUT_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "../results"))
QUALITY_OUT = os.path.join(OUTPUT_DIR, "validation_quality_score_384.csv")
COMMIT_RAW_OUT = os.path.join(OUTPUT_DIR, "validation_commit_messages_raw.json")
COMMIT_OUT = os.path.join(OUTPUT_DIR, "validation_commit_intent_384.csv")
PROVENANCE_OUT = os.path.join(OUTPUT_DIR, "validation_commit_provenance.json")

SAMPLE_SIZE = 384
RANDOM_SEED = 2025
N_SATD_INSTANCES = 80  # oversample to account for API failures


# ===================================================================
# Phase 1: quality_score — prepare annotation CSV from existing sample
# ===================================================================

def phase_quality_score():
    """Extract quality_score annotation CSV from existing 384-sample data."""
    print("=" * 60)
    print("Phase 1: Preparing quality_score annotation CSV")
    print("=" * 60)

    with open(SAMPLE_384_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"  Loaded {len(rows)} records from existing 384-sample CSV")

    # Only keep confirmed SATD instances (manual_is_satd == '1')
    valid_rows = [r for r in rows if r.get("manual_is_satd") == "1"]
    print(f"  Confirmed SATD instances (manual_is_satd=1): {len(valid_rows)}")

    # LLM label distribution
    dist = Counter(r["satd_quality_score"] for r in valid_rows)
    print(f"  LLM quality_score distribution: {dict(sorted(dist.items()))}")
    print(f"    1 (vague): {dist.get('1', 0)}")
    print(f"    2 (issue-identifying): {dist.get('2', 0)}")
    print(f"    3 (actionable): {dist.get('3', 0)}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(QUALITY_OUT, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "sample_id",
            "satd_id",
            "project_name",
            "f_comment",
            "llm_quality_score",
            "annotator_A",
            "annotator_B",
            "annotator_C",
            "gold_label",
            "notes",
        ])
        for i, r in enumerate(valid_rows, 1):
            writer.writerow([
                i,
                r["satd_id"],
                r["project_name"],
                r["f_comment"],
                r["satd_quality_score"],
                "",  # annotator A
                "",  # annotator B
                "",  # annotator C (tiebreaker)
                "",  # gold label (computed after annotation)
                "",  # notes
            ])

    print(f"\n  Output: {QUALITY_OUT}")
    print(f"  Total annotation units: {len(valid_rows)}")
    print("  Phase 1 complete.\n")
    return len(valid_rows)


# ===================================================================
# Phase 2: past_*_ratio — retrieve exact historical commits and classify
# ===================================================================

def _github_api_get(url, max_retries=3):
    """Make an authenticated GitHub API GET request with rate-limit handling."""
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "SATD-Validation-Script/1.0",
    }
    token = os.environ.get("GITHUB_TOKEN", "")
    if token:
        headers["Authorization"] = f"token {token}"

    for attempt in range(max_retries):
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                remaining = resp.headers.get("X-RateLimit-Remaining", "?")
                body = json.loads(resp.read().decode("utf-8"))
                return body, int(remaining) if remaining != "?" else 999
        except urllib.error.HTTPError as e:
            if e.code == 403:
                reset_ts = e.headers.get("X-RateLimit-Reset", "")
                wait = max(int(reset_ts) - int(time.time()), 60) if reset_ts else 60
                print(f"    [RATE LIMITED] waiting {wait}s (attempt {attempt+1})...")
                time.sleep(wait + 2)
                continue
            if e.code == 409:
                return [], 999
            print(f"    [HTTP {e.code}] {url}: {e.reason}")
            return [], 999
        except Exception as e:
            print(f"    [ERROR] {url}: {e}")
            if attempt < max_retries - 1:
                time.sleep(3)
    return [], 999


def fetch_commits_for_instance(owner, repo, author_email, before_date_iso, n=10):
    """Fetch prior commits for one SATD instance — mirrors original git-log call."""
    url = (
        f"https://api.github.com/repos/{owner}/{repo}/commits"
        f"?author={urllib.request.quote(author_email)}"
        f"&until={before_date_iso}"
        f"&per_page={n}"
    )
    data, remaining = _github_api_get(url)
    if not isinstance(data, list):
        return [], remaining

    commits = []
    for item in data[:n]:
        sha = item["sha"]
        msg = item["commit"]["message"].split("\n")[0].strip()
        if msg:
            commits.append({"commit_hash": sha, "commit_message": msg})
    return commits, remaining


def classify_commit_deepseek(commit_message):
    """Classify a commit message using the identical prompt as the original pipeline."""
    try:
        from openai import OpenAI
    except ImportError:
        print("  ERROR: openai package not installed. Run: pip install openai")
        sys.exit(1)

    api_key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("  ERROR: set OPENAI_API_KEY or DEEPSEEK_API_KEY before DeepSeek-V3 classification.")
        sys.exit(1)
    api_base = os.environ.get(
        "DEEPSEEK_API_BASE",
        os.environ.get(
            "OPENAI_API_BASE",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        ),
    )

    client = OpenAI(base_url=api_base, api_key=api_key)

    prompt_body = (
        "You are an expert code-review assistant."
        "Your task is to read a single git commit message and decide **its primary purpose**."
        "Choose exactly one tag from the list below:"
        "1. <feature>   — Adds new functionality or user-visible feature\n"
        "2. <bugfix>    — Fixes a bug, error, or incorrect behavior\n"
        "3. <refactor>  — Refactors or restructures code without changing behavior\n"
        "4. <cleanup>   — Removes dead code, comments, or performs non-functional cleanup\n"
        "5. <other>     — Anything else (build scripts, config, dependency bumps, etc.)"
        "\nOutput only the tag.only the tag."
        "\nAnalyze carefully."
        "\n\n"
    )
    prompt_examples = (
        "Example1:\n"
        "Commit Message:\n"
        "`Add OAuth2 login flow with Google provider`\n"
        "Response: `<feature>`\n\n"
        "Commit Message:\n"
        "`Fix null-pointer exception in UserService when email is missing`\n"
        "Response: `<bugfix>`\n\n"
        "Commit Message:\n"
        "`Refactor OrderController to use service layer abstraction`\n"
        "Response: `<refactor>`\n\n"
        "Commit Message:\n"
        "`Remove unused helper methods and obsolete classes`\n"
        "Response: `<cleanup>`\n\n"
        "Commit Message:\n"
        "`Bump dependency versions and warnings`\n"
        "Response: `<other>`\n\n"
    )
    prompt = prompt_body + prompt_examples + f"\nCommit Message:\n{commit_message}\nResponse:"

    try:
        response = client.chat.completions.create(
            model="deepseek-v3",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        raw = response.choices[0].message.content.strip()
        return _extract_tag(raw), False
    except Exception as e:
        print(f"    [LLM ERROR] {e}")
        time.sleep(5)
        return "other", True


def _extract_tag(response):
    """Extract tag from LLM response (same logic as original pipeline)."""
    allowed = {"feature", "bugfix", "refactor", "cleanup", "docs", "perf", "test", "other"}
    matches = re.findall(r"<([^>]+)>", response)
    if matches:
        last = matches[-1].strip()
        return last if last in allowed else "other"
    return "other"


def _fetch_commits_from_satd_instances():
    """Retrieve historical commits matching original pipeline parameters via GitHub API."""

    with open(RAW_DATA_JSON, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    with open(MERGED_DATA_JSON, "r", encoding="utf-8") as f:
        merged_data = json.load(f)

    ratio_keys = [
        "developer_past_bugfix_ratio", "developer_past_cleanup_ratio",
        "developer_past_feature_ratio", "developer_past_refactor_ratio",
    ]
    has_commits = {
        r["satd_id"] for r in merged_data
        if any(r.get(k, 0) > 0 for k in ratio_keys)
    }
    candidates = [r for r in raw_data if r["satd_id"] in has_commits]
    print(f"  SATD instances with commit history: {len(candidates)} / {len(raw_data)}")

    random.seed(RANDOM_SEED)
    n_sample = min(N_SATD_INSTANCES, len(candidates))
    sampled_instances = random.sample(candidates, n_sample)
    print(f"  Sampled {n_sample} SATD instances for commit retrieval")

    all_commits = []
    seen_hashes = set()
    provenance = []

    for idx, inst in enumerate(sampled_instances, 1):
        satd_id = inst["satd_id"]
        project = inst["project_name"]
        email = inst["adder_email"]
        add_date = inst["add_date"]

        iso_date = add_date.replace(" ", "T") + "Z"
        parts = project.split("/")
        if len(parts) != 2:
            continue
        owner, repo = parts

        print(f"  [{idx}/{n_sample}] satd_id={satd_id}  {project}  "
              f"author={email}  before={add_date}  ...", end=" ")

        commits, remaining = fetch_commits_for_instance(
            owner, repo, email, iso_date, n=10
        )

        added = 0
        instance_hashes = []
        for c in commits:
            instance_hashes.append(c["commit_hash"])
            if c["commit_hash"] not in seen_hashes:
                seen_hashes.add(c["commit_hash"])
                c["project_name"] = project
                c["source_satd_id"] = satd_id
                c["source_adder_email"] = email
                c["source_add_date"] = add_date
                all_commits.append(c)
                added += 1

        provenance.append({
            "satd_id": satd_id,
            "project_name": project,
            "adder_email": email,
            "add_date": add_date,
            "api_url": (
                f"https://api.github.com/repos/{owner}/{repo}/commits"
                f"?author={email}&until={iso_date}&per_page=10"
            ),
            "commits_returned": len(commits),
            "commits_new": added,
            "commit_hashes": instance_hashes,
        })

        print(f"got {len(commits)} commits ({added} new)  [API remaining: {remaining}]")

        if remaining <= 5:
            print("  WARNING: API rate limit nearly exhausted, stopping early")
            break

        time.sleep(1.0)

    print(f"\n  Total unique commit messages fetched: {len(all_commits)}")
    print(f"  From {len(provenance)} SATD instances")

    if len(all_commits) < SAMPLE_SIZE:
        print(f"  WARNING: Only {len(all_commits)} commits, need {SAMPLE_SIZE}")
        print("  TIP: Set GITHUB_TOKEN env var for 5000 req/hr rate limit")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(COMMIT_RAW_OUT, "w", encoding="utf-8") as f:
        json.dump(all_commits, f, indent=2, ensure_ascii=False)
    print(f"  Saved raw commits to: {COMMIT_RAW_OUT}")

    with open(PROVENANCE_OUT, "w", encoding="utf-8") as f:
        json.dump({
            "description": (
                "Provenance manifest for past_*_ratio validation. "
                "Each entry records the SATD instance sampled and the "
                "GitHub Commits API parameters used, mirroring the original "
                "git-log --author={email} --before={add_date} -n10 pipeline."
            ),
            "random_seed": RANDOM_SEED,
            "n_instances_sampled": len(provenance),
            "n_unique_commits": len(all_commits),
            "generated_at": datetime.now().isoformat(),
            "instances": provenance,
        }, f, indent=2, ensure_ascii=False)
    print(f"  Saved provenance manifest to: {PROVENANCE_OUT}")

    return all_commits


def _save_checkpoint(all_commits):
    """Incrementally save classified commits to avoid losing progress."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(COMMIT_RAW_OUT, "w", encoding="utf-8") as f:
        json.dump(all_commits, f, indent=2, ensure_ascii=False)


def phase_commit_intent(skip_classify=False):
    """Retrieve historical commits and classify for human validation."""
    print("=" * 60)
    print("Phase 2: Preparing commit intent (past_*_ratio) annotation CSV")
    print("=" * 60)

    if os.path.exists(COMMIT_RAW_OUT):
        print(f"  Found cached commits: {COMMIT_RAW_OUT}")
        with open(COMMIT_RAW_OUT, "r", encoding="utf-8") as f:
            all_commits = json.load(f)
        print(f"  Loaded {len(all_commits)} cached commit messages, skipping GitHub API")
    else:
        all_commits = _fetch_commits_from_satd_instances()

    if not all_commits:
        print("  ERROR: No commit messages available. Aborting Phase 2.")
        return 0

    if skip_classify:
        unclassified = sum(1 for c in all_commits if "llm_label" not in c)
        print(f"\n  --skip-classify: {len(all_commits)} commits fetched, "
              f"{unclassified} unclassified.")
        print("  Re-run without --skip-classify after setting a valid API key.")
        return len(all_commits)

    need_classify = [c for c in all_commits if "llm_label" not in c]
    if need_classify:
        print(f"\n  Classifying {len(need_classify)} commit messages with DeepSeek-V3...")
        classify_cache = {}
        consecutive_api_errors = 0
        for idx, c in enumerate(need_classify, 1):
            h = c["commit_hash"]
            msg = c["commit_message"]
            if h in classify_cache:
                c["llm_label"] = classify_cache[h]
            else:
                label, was_error = classify_commit_deepseek(msg)
                c["llm_label"] = label
                classify_cache[h] = label
                if was_error:
                    consecutive_api_errors += 1
                else:
                    consecutive_api_errors = 0
                if consecutive_api_errors >= 5:
                    print(f"\n  ABORT: 5 consecutive API errors — likely out of quota.")
                    print(f"  Saving checkpoint with {idx} items processed...")
                    _save_checkpoint(all_commits)
                    classified_total = sum(1 for x in all_commits if "llm_label" in x)
                    print(f"  Classified so far: {classified_total}/{len(all_commits)}")
                    print(f"  Fix API key/balance and re-run to continue from checkpoint.")
                    return 0
                if idx % 50 == 0:
                    print(f"    Classified {idx}/{len(need_classify)}...")
                    _save_checkpoint(all_commits)
                time.sleep(0.3)

        _save_checkpoint(all_commits)
        print(f"  Saved classified commits to: {COMMIT_RAW_OUT}")
    else:
        print(f"  All {len(all_commits)} commits already classified, skipping LLM step")

    label_dist = Counter(c["llm_label"] for c in all_commits)
    print(f"\n  LLM label distribution (all {len(all_commits)} commits):")
    for label, count in sorted(label_dist.items()):
        print(f"    {label}: {count}")

    random.seed(RANDOM_SEED)
    by_label = {}
    for c in all_commits:
        by_label.setdefault(c["llm_label"], []).append(c)

    total = len(all_commits)
    sampled_commits = []
    target = min(SAMPLE_SIZE, total)

    for label, items in sorted(by_label.items()):
        n_alloc = max(5, round(len(items) / total * target))
        n_alloc = min(n_alloc, len(items))
        sampled_commits.extend(random.sample(items, n_alloc))

    if len(sampled_commits) > target:
        sampled_commits = random.sample(sampled_commits, target)
    elif len(sampled_commits) < target:
        remaining_pool = [c for c in all_commits if c not in sampled_commits]
        extra = random.sample(
            remaining_pool, min(target - len(sampled_commits), len(remaining_pool))
        )
        sampled_commits.extend(extra)

    random.shuffle(sampled_commits)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(COMMIT_OUT, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "sample_id",
            "commit_hash",
            "project_name",
            "source_satd_id",
            "source_adder_email",
            "source_add_date",
            "commit_message",
            "llm_label",
            "annotator_A",
            "annotator_B",
            "annotator_C",
            "gold_label",
            "notes",
        ])
        for i, c in enumerate(sampled_commits, 1):
            writer.writerow([
                i,
                c["commit_hash"],
                c["project_name"],
                c.get("source_satd_id", ""),
                c.get("source_adder_email", ""),
                c.get("source_add_date", ""),
                c["commit_message"],
                c["llm_label"],
                "", "", "", "", "",
            ])

    final_dist = Counter(c["llm_label"] for c in sampled_commits)
    print(f"\n  Final sample distribution ({len(sampled_commits)} commits):")
    for label, count in sorted(final_dist.items()):
        print(f"    {label}: {count}")

    print(f"\n  Output: {COMMIT_OUT}")
    print("  Phase 2 complete.\n")
    return len(sampled_commits)


# ===================================================================
# Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser(description="Sample LLM features for human validation")
    parser.add_argument(
        "--phase",
        choices=["quality", "commit", "all"],
        default="all",
        help="Which phase to run: quality (quality_score only), commit (past_*_ratio only), all (both)",
    )
    parser.add_argument(
        "--skip-classify",
        action="store_true",
        help="Fetch commits only, skip DeepSeek-V3 classification (use when API key is unavailable)",
    )
    args = parser.parse_args()

    print(f"Random seed: {RANDOM_SEED}")
    print(f"Target sample size: {SAMPLE_SIZE}")
    print(f"Output directory: {OUTPUT_DIR}\n")

    if args.phase in ("quality", "all"):
        n = phase_quality_score()
        print(f"  quality_score: {n} annotation units ready\n")

    if args.phase in ("commit", "all"):
        n = phase_commit_intent(skip_classify=args.skip_classify)
        print(f"  commit_intent: {n} annotation units ready\n")

    print("Done. Next steps:")
    if args.skip_classify:
        print("  1. Set a valid OPENAI_API_KEY (DashScope) and re-run without --skip-classify")
    else:
        print("  1. Distribute CSV files to annotators with annotation guidelines")
        print("  2. After annotation, run 23_compute_llm_validation_metrics.py")


if __name__ == "__main__":
    main()
