import os
import json
import git
from datetime import datetime, timezone
import math
import sys
from collections import defaultdict

def calculate_active_days(raw_data_path, output_dir, repo_base_dir):
    """
    Calculate the number of active days of the developer in the project for each SATD record.
    """
    with open(raw_data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    features = []
    # Cache: project name -> {email: earliest commit time, name: earliest commit time}
    email_cache = defaultdict(dict) 
    name_cache = defaultdict(dict)   
    
    for entry in data:
        print("-"*60)
        satd_id = entry['satd_id']
        project_name = entry['project_name']
        adder_email = entry.get('adder_email', '').strip()
        adder_name = entry.get('adder_name', '').strip()
        add_date = entry['add_date']
        
        # Parse add_date and convert to UTC timestamp
        add_date_naive = datetime.strptime(add_date, "%Y-%m-%d %H:%M:%S")
        add_date_utc = add_date_naive.replace(tzinfo=timezone.utc)
        add_date_ts = add_date_utc.timestamp()
        
        # Build project repository path
        repo_path = os.path.join(repo_base_dir, project_name)
        
        active_days = 0
        Found = False # Whether found in cache
        first_commit_ts = None
        
        # Check email cache
        if adder_email and project_name in email_cache and adder_email in email_cache[project_name]:
            email_ts = email_cache[project_name][adder_email]
            if first_commit_ts is None or email_ts < first_commit_ts:
                first_commit_ts = email_ts
            Found = True
        
        # Check name cache
        if adder_name and project_name in name_cache and adder_name in name_cache[project_name]:
            name_ts = name_cache[project_name][adder_name]
            if first_commit_ts is None or name_ts < first_commit_ts:
                first_commit_ts = name_ts
            Found = True
        
        # If timestamp found in cache, calculate active days
        if Found:
            time_diff_seconds = add_date_ts - first_commit_ts
            active_days = math.ceil(time_diff_seconds / (60 * 60 * 24)) if time_diff_seconds >= 0 else 0
            
            features.append({
                'satd_id': satd_id,
                'project_name': project_name,
                'developer_active_days': active_days
            })
            print(f"add_date: {add_date_ts}, first_commit_ts: {first_commit_ts}")
            print(f"[Cache] SATD {satd_id} developer {adder_email}/{adder_name} active time in project {project_name}: {active_days} days")
            continue
        
        repo = git.Repo(repo_path)
        
        # Get all commits by the developer in this project, match by email or name
        commits = []
        for commit in repo.iter_commits():
            author = commit.author
            if (adder_email and author.email == adder_email) or (adder_name and author.name == adder_name):
                commits.append(commit)
        
        if not commits:
            print(f"Warning: Developer {adder_email}/{adder_name} has no commit records in project {project_name}")
            features.append({
                'satd_id': satd_id,
                'project_name': project_name,
                'developer_active_days': active_days
            })
            continue
        
        # Find the earliest commit time
        first_commit_ts = min(commit.committed_date for commit in commits)
        
        # Update cache - use both email and name as keys
        if adder_email:
            if adder_email not in email_cache[project_name] or first_commit_ts < email_cache[project_name][adder_email]:
                email_cache[project_name][adder_email] = first_commit_ts
        
        if adder_name:
            if adder_name not in name_cache[project_name] or first_commit_ts < name_cache[project_name][adder_name]:
                name_cache[project_name][adder_name] = first_commit_ts
        
        # Calculate days difference (round up)
        time_diff_seconds = add_date_ts - first_commit_ts
        active_days = math.ceil(time_diff_seconds / (60 * 60 * 24)) if time_diff_seconds >= 0 else 0
        
        # Store result
        features.append({
            'satd_id': satd_id,
            'project_name': project_name,
            'developer_active_days': active_days
        })
        print(f"add_date: {add_date_ts}, first_commit_ts: {first_commit_ts}")
        print(f"SATD {satd_id} developer {adder_email}/{adder_name} active time in project {project_name}: {active_days} days")
    
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

    calculate_active_days(RAW_DATA_PATH, FEATURES_DIR, REPO_BASE_DIR)