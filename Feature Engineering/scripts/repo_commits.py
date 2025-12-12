import os
import sys
import json
import subprocess
from datetime import datetime, timedelta


def analyze_project_commits(input_path, output_dir, repo_base_dir):
    with open(input_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    features = []
      
    for record in raw_data:
        satd_id = record["satd_id"]
        project_name = record["project_name"]
        target_date = datetime.strptime(record["add_date"], "%Y-%m-%d %H:%M:%S")
        active_period_start = target_date - timedelta(days=180)
        
        # Construct repository path
        repo_path = os.path.join(repo_base_dir, project_name)
        
        # Convert date format to Git-recognizable format
        target_date_str = target_date.strftime("%Y-%m-%d %H:%M:%S")
        active_start_str = active_period_start.strftime("%Y-%m-%d %H:%M:%S")
        
        # Get total commit count
        cmd_total = [
            'git', '-C', repo_path, 'log', 
            '--before', target_date_str, 
            '--pretty=format:%H'
        ]
        
        result = subprocess.run(cmd_total, capture_output=True, text=True, check=True)
        total_commits = len(result.stdout.splitlines())
        
        # Get active commit count
        cmd_active = [
            'git', '-C', repo_path, 'log', 
            '--after', active_start_str,
            '--before', target_date_str,
            '--pretty=format:%H'
        ]

        result = subprocess.run(cmd_active, capture_output=True, text=True, check=True)
        active_commits = len(result.stdout.splitlines())

        
        # Construct feature record
        feature_record = {
            "satd_id": satd_id,
            "project_name": project_name,
            "project_total_commits": total_commits,
            "project_active_commits": active_commits
        }
        features.append(feature_record)
        print(f"Project {project_name} for satd_id {satd_id}: total commits {total_commits}, active commits {active_commits}")
    
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
    
    analyze_project_commits(RAW_DATA_PATH, FEATURES_DIR, REPO_BASE_DIR)