import os
import json
import math
import subprocess
from datetime import datetime, timezone

def get_last_commit_days(repo_path, commit_hash, add_date, adder_name, adder_email):
    """
    Get the number of days between the developer's previous commit before commit_hash and add_date
    For name and email, calculate the days difference separately, then take the maximum value
    """
    add_dt = datetime.strptime(add_date, '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc)

    # Get the Unix timestamp of the previous commit filtered by author
    def get_last_commit_date(author):
        cmd = ['git', '-C', repo_path, 'log', '-1', '--pretty=format:%at', f'--author={author}', f'{commit_hash}^']
        try:
            res = subprocess.run(cmd, check=True, capture_output=True, text=True, encoding='utf-8', errors='ignore')
            ts = res.stdout.strip()
            return int(ts) if ts.isdigit() else None
        except subprocess.CalledProcessError:
            return None

    # Query previous commit time for name and email separately
    commit_date_by_name = get_last_commit_date(adder_name)
    commit_date_by_email = get_last_commit_date(adder_email)


    # Calculate days difference between each time and add_date
    def calculate_days(ts):
        if ts is None:
            return 0
        last_commit_dt = datetime.fromtimestamp(ts, timezone.utc)
        delta = add_dt - last_commit_dt
        return math.ceil(delta.total_seconds() / (24 * 3600))


    days_by_name = calculate_days(commit_date_by_name)
    days_by_email = calculate_days(commit_date_by_email)
    
    # If both queries fail, report error
    if commit_date_by_name is None and commit_date_by_email is None:
        print(f"No previous commits found for {adder_name} or {adder_email} before {commit_hash}")
        return 0
    
    # Take the maximum value of the two days differences
    return max(days_by_name, days_by_email, 0)



def process_raw_data(input_json: str, output_dir: str, repo_base_dir: str):
    with open(input_json, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    features = []
    
    for rec in raw_data:
        print("-" * 60)
        satd_id = rec.get('satd_id')
        project_name = rec.get('project_name')
        adder_name = rec.get('adder_name')
        adder_email = rec.get('adder_email')
        add_hash = rec.get('add_commit_hash')
        add_date = rec.get('add_date')
        
        developer_last_commit_days = 0
        # Construct repository path
        repo_path = os.path.join(repo_base_dir, project_name)
        
        developer_last_commit_days = get_last_commit_days(repo_path, add_hash, add_date, adder_name, adder_email)
        
        features.append({
            'satd_id': satd_id,
			'project_name': project_name,
			'developer_last_commit_days': developer_last_commit_days,
		})
        print(f"satd_id: {satd_id}, developer_last_commit_days: {developer_last_commit_days}")
    
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

    process_raw_data(RAW_DATA_PATH, FEATURES_DIR, REPO_BASE_DIR)
