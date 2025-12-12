import os
import json
import subprocess
import subprocess
from datetime import datetime, timezone
import math

def get_last_commit_days(repo_path, commit_hash, add_date):
    """
    Get the previous commit (parent commit) of commit_hash, and calculate the number of days from add_date to that parent commit.
    """

    # Parse the input add_date string as a datetime object (assumed to be local time, specify UTC manually if needed)
    add_date = datetime.strptime(add_date, '%Y-%m-%d %H:%M:%S')

    # Get the author timestamp of the parent commit
    cmd_father_date = [
        'git', '-C', repo_path, 'log', '-1', '--pretty=format:%at', f"{commit_hash}^"
    ]
    
    result = subprocess.run(cmd_father_date, capture_output=True, text=True, encoding='utf-8', errors='replace')

    parent_commit_timestamp = result.stdout.strip()

    if not parent_commit_timestamp:
        print(f"No parent commit found for {commit_hash} in {repo_path}")
        return 0

    parent_commit_date = datetime.fromtimestamp(int(parent_commit_timestamp), tz=timezone.utc)
    
    add_date_utc = add_date.replace(tzinfo=timezone.utc)  

    # Calculate days difference
    time_diff = add_date_utc - parent_commit_date
    time_diff_seconds = time_diff.total_seconds()
    days = math.ceil(time_diff_seconds / (60 * 60 * 24)) 

    return max(days, 0)

def process_raw_data(input_json, output_dir, repo_base_dir):
    with open(input_json, 'r', encoding='utf-8') as f:
        records = json.load(f)
        
    features = []
    for rec in records:
        print("-" * 60)
        satd_id = rec.get('satd_id')
        project_name = rec.get('project_name')
        add_hash = rec.get('add_commit_hash')
        add_date = rec.get('add_date', '').strip()

        repo_last_commit_days = 0
        # Construct repository path
        repo_path = os.path.join(repo_base_dir, project_name)

        repo_last_commit_days = get_last_commit_days(repo_path, add_hash, add_date)
        
        features.append({
            'satd_id': satd_id,
			'project_name': project_name,
			'project_last_commit_days': repo_last_commit_days,
		})
        
        print(f"[{satd_id}] Processed {project_name} @ {add_hash} -> {repo_last_commit_days} days")

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
        os.path.join(current_dir, "../../Dataset/repos/")
    )
    process_raw_data(RAW_DATA_PATH, FEATURES_DIR, REPO_BASE_DIR)