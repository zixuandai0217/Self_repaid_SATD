import os
import json
import git
import sys
from datetime import datetime, timezone


def blame_ownership(repo, rev, f_path, adder_email, adder_name):
    """
    Execute git blame on the file at the specified version, calculate the number of lines by the specified author and total lines, return the code line ratio
    """
    try:
        blame_data = repo.blame(rev, f_path)
    except Exception as e:
        print(f"Warning: Unable to execute blame on {f_path}@{rev} - {e}")
        return 0.0

    total_lines = 0
    owned_lines = 0
    for commit, lines in blame_data:
        count = len(lines)
        total_lines += count
        author = commit.author
        if (adder_email and author.email == adder_email) or (adder_name and author.name == adder_name):
            owned_lines += count
    return round(owned_lines / total_lines, 4) if total_lines > 0 else 0.0


def calculate_ownership_features(raw_data_path, output_dir, repo_base_dir):
    """
	calculate_ownership_features.py

	Calculate code ownership features at the time of SATD introduction:
		- developer_ownership: Ownership based on the commit when SATD was introduced (add_commit_hash)
		- By default, for records where information cannot be obtained, ownership features are set to 0.0
	"""
    with open(raw_data_path, 'r', encoding='utf-8') as f:
        satd_data = json.load(f)

    features = []

    for entry in satd_data:
        print('-'*60)
        satd_id     = entry.get('satd_id')
        project     = entry.get('project_name', '').strip()
        adder_email = entry.get('adder_email', '').strip()
        adder_name  = entry.get('adder_name', '').strip()
        file_rel    = entry.get('f_path', '').lstrip('/')
        add_commit_hash= entry.get('add_commit_hash', '').strip()

        # Default features
        feat = {
            'satd_id': satd_id,
            'project_name': project,
            'developer_ownership': 0.0
        }

        if not (project and file_rel and (adder_email or adder_name) and add_commit_hash):
            print(f"Warning: SATD {satd_id} missing necessary information or add_commit_hash, using default ownership 0.0")
            features.append(feat)
            continue

        # Open repository
        repo_path = os.path.join(repo_base_dir, project.replace('/', os.path.sep))
        repo = git.Repo(repo_path)

        # Calculate ownership at introduction time
        ratio = blame_ownership(repo, add_commit_hash, file_rel, adder_email, adder_name)

        # Update features
        feat['developer_ownership'] = ratio
        features.append(feat)
        print(f"Processed SATD {satd_id}: ownership = {ratio}")

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
    
    calculate_ownership_features(RAW_DATA_PATH, FEATURES_DIR, REPO_BASE_DIR)
