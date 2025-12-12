import os
import sys
import json
import subprocess
from datetime import datetime, timedelta

def is_same_developer(developer1, developer2):
    """Determine if two developers are the same person (same if name or email matches)"""
    name1, email1 = developer1
    name2, email2 = developer2
    return name1== name2 or email1 == email2


def get_developers(cmd):
	result = subprocess.run(cmd,capture_output=True,check=True,encoding='utf-8',errors='replace')
	
	developers = set()
	for line in result.stdout.splitlines():
		line = line.strip()
		if not line or '|' not in line:
			continue
		results = line.split('|', 1)
		if len(results) != 2:
			continue
		name, email = results
		developer = (name.strip(), email.strip())
		
		is_exist = any(is_same_developer(developer, existing_developer) for existing_developer in developers)
		if not is_exist:
			developers.add(developer)
                  
	return len(developers)

def analyze_project_developers(input_path, output_dir, repo_base_dir):
    with open(input_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    features = []
    for record in raw_data:
        print('-' * 60)
        satd_id = record["satd_id"]
        project_name = record["project_name"]
        target_date = datetime.strptime(record["add_date"], "%Y-%m-%d %H:%M:%S")
        active_period_start = target_date - timedelta(days=180)
        
        # Construct repository path
        repo_path = os.path.join(repo_base_dir, project_name)
        
        # Convert date format to Git-recognizable format
        target_date_str = target_date.strftime("%Y-%m-%d %H:%M:%S")
        active_start_str = active_period_start.strftime("%Y-%m-%d %H:%M:%S")

        # Get all developers
        cmd_total_developers = [
            'git', '-C', repo_path, 'log', 
            '--before', target_date_str,
            '--pretty=format:%an|%ae'  # Author name|Author email
        ]
        
        # Get active developers
        cmd_active_developers = [
            'git', '-C', repo_path, 'log', 
            '--after', active_start_str,
            '--before', target_date_str,
            '--pretty=format:%an|%ae' # Author name|Author email
        ]
        
        all_developers = get_developers(cmd_total_developers)
        active_developers = get_developers(cmd_active_developers)

        features.append({
			'satd_id': satd_id,
			'project_name': project_name,
			'project_total_developers': all_developers,
			'project_active_developers': active_developers
		})
        print(f"Processing completed: satd_id {satd_id}, project {project_name}, total developers {all_developers}, active developers {active_developers}")

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
    
    analyze_project_developers(RAW_DATA_PATH, FEATURES_DIR, REPO_BASE_DIR)