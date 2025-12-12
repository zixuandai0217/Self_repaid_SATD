import os
import json
import git
from datetime import datetime, timezone
import math
import sys

def calculate_project_active_days(input_path, output_dir, repo_base_dir):
    """
    Calculate the number of days from the first commit to SATD add_date for each SATD record's project
    and cache the first commit time by project_name
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        satd_list = json.load(f)

    features = []
    first_commit_cache = {}

    for entry in satd_list:
        print("-" * 60)
        satd_id = entry.get('satd_id')
        project_name = entry.get('project_name', '').strip()
        add_date = entry.get('add_date', '').strip()

        project_active_days = 0
        # Parse add_date as UTC timestamp

        dt_naive = datetime.strptime(add_date, "%Y-%m-%d %H:%M:%S")
        dt_utc   = dt_naive.replace(tzinfo=timezone.utc)
        add_ts   = dt_utc.timestamp()

        repo_path = os.path.join(repo_base_dir, *project_name.split('/'))
        
        # Check cache
        first_ts = first_commit_cache.get(project_name)
        if not first_ts:
            repo = git.Repo(repo_path)
            commits = list(repo.iter_commits(all=True))
            first_commit = min(commits, key=lambda c: c.committed_date)
            first_ts = first_commit.committed_date
            first_commit_cache[project_name] = first_ts

        # Calculate active days
        diff = add_ts - first_ts
        project_active_days = math.ceil(diff / (24*3600)) if diff >= 0 else 0

        features.append({
            'satd_id': satd_id,
            'project_name': project_name,
            'project_active_days': project_active_days
        })
        print(f"Project {project_name} active days: {project_active_days}")

    script_name = os.path.splitext(os.path.basename(__file__))[0]
    output_path = os.path.join(output_dir, f"{script_name}.json")
    
    os.makedirs(output_dir, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(features, f, indent=4, ensure_ascii=False)
    print(f"Feature file successfully generated: {output_path}")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    RAW_DATA_PATH = os.path.normpath(
        os.path.join(current_dir, "../../Dataset/data/raw_data_final_40501.json")
    )
    FEATURES_DIR = os.path.normpath(
        os.path.join(current_dir, "../features/")
    )
    REPO_BASE_DIR = os.path.normpath(
        os.path.join(current_dir,"../../Dataset/repos/")
    )
    
    calculate_project_active_days(RAW_DATA_PATH, FEATURES_DIR, REPO_BASE_DIR)
