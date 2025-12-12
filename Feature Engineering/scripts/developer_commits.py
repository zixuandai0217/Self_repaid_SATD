import os
import json
import subprocess
from datetime import datetime, timedelta

def get_developer_total_commits(repo_path, target_date_str, name, email):
	# Author: dzx0217 <dzx0217@qq.com>
	# Query author name
	cmd_name = [
		'git', '-C', repo_path, 'log',
		'--before', target_date_str,
		'--author', f'{name} ',
		'--pretty=format:%H'
	]
	 
	# Query author email
	cmd_email = [
		'git', '-C', repo_path, 'log',
		'--before', target_date_str,
		'--author', f'<{email}>',
		'--pretty=format:%H'
	]
	 
	result_name = subprocess.run(cmd_name, capture_output=True, text=True, check=True)
	commits_name = set(result_name.stdout.strip().splitlines())
	 
	result_email = subprocess.run(cmd_email, capture_output=True, text=True, check=True)
	commits_email = set(result_email.stdout.strip().splitlines())
	 
	# Merge two sets to ensure no duplicates
	all_commits = commits_name.union(commits_email)
	return len(all_commits)

def get_developer_active_commits(repo_path, active_start_str, target_date_str, name, email):
	# Query author name
	cmd_name = [
		'git', '-C', repo_path, 'log',
		'--after', active_start_str,
		'--before', target_date_str,
		'--author', f'{name} ',
		'--pretty=format:%H'
	]
	 
	# Query author email
	cmd_email = [
		'git', '-C', repo_path, 'log',
		'--after', active_start_str,
		'--before', target_date_str,
		'--author', f'<{email}>',
		'--pretty=format:%H'
	]
	 
	result_name = subprocess.run(cmd_name, capture_output=True, text=True, check=True)
	commits_name = set(result_name.stdout.strip().splitlines())
	 
	result_email = subprocess.run(cmd_email, capture_output=True, text=True, check=True)
	commits_email = set(result_email.stdout.strip().splitlines())
     
	# Merge two sets to ensure no duplicates
	all_commits = commits_name.union(commits_email)
	return len(all_commits)

def calculate_developer_commits(raw_data_path, output_dir, repo_base_dir):
    """
    Calculate the historical commit count of the developer in the project before each SATD is added (add_date).
    """
    with open(raw_data_path, 'r', encoding='utf-8') as f:
        satd_data = json.load(f)
        
    features = []
    for entry in satd_data:
        print("-" * 60)
        satd_id      = entry.get('satd_id')
        project_name = entry.get('project_name', '').strip()
        adder_email  = entry.get('adder_email', '').strip()
        adder_name   = entry.get('adder_name', '').strip()
        add_date     = entry.get('add_date', '').strip()
        target_date = datetime.strptime(add_date, "%Y-%m-%d %H:%M:%S")
        active_period_start = target_date - timedelta(days=180)

        repo_path = os.path.join(repo_base_dir, project_name)
        
        # Convert date format to Git-recognizable format
        target_date_str = target_date.strftime("%Y-%m-%d %H:%M:%S")
        active_start_str = active_period_start.strftime("%Y-%m-%d %H:%M:%S")

        # Get developer total commit count
        developer_total_commits = get_developer_total_commits(repo_path, target_date_str, adder_name, adder_email)
        
        # Get developer commit count in last 180 days
        developer_active_commits = get_developer_active_commits(repo_path, active_start_str, target_date_str, adder_name, adder_email)

        features.append({
			'satd_id': satd_id,
			'project_name': project_name,
			'developer_total_commits': developer_total_commits,
			'developer_active_commits': developer_active_commits
		})
        
        print(f"Processed SATD ID: {satd_id}, Project: {project_name}\n"
              f"{adder_name}@{adder_email} --> Total Commits: {developer_total_commits}, Active Commits: {developer_active_commits}")

    script_name = os.path.splitext(os.path.basename(__file__))[0]
    output_path = os.path.join(output_dir, f"{script_name}.json")
    
    os.makedirs(output_dir, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(features, f, indent=4, ensure_ascii=False)
    print(f"Feature file successfully generated: {output_path}")


if __name__ == '__main__':
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

    calculate_developer_commits(RAW_DATA_PATH, FEATURES_DIR, REPO_BASE_DIR)
