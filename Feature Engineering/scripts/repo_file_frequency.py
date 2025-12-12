import os
import json
import git
import sys

def is_same_developer(developer1, developer2):
    """Determine if two developers are the same person (same if name or email matches)"""
    name1, email1 = developer1
    name2, email2 = developer2
    return name1== name2 or email1 == email2

def get_file_frequency(repo, rev, f_path):
    """
    Calculate the number of modifications and participating developers for path f_path before (including) commit rev.
    """
    try:
        # repo.iter_commits will list all commits in the history reachable from rev that involve f_path
        commits = list(repo.iter_commits(rev, paths=f_path))
    except Exception as e:
        print(f"Warning: Unable to list commits for {f_path}@{rev} - {e}")
        return 0, 0

    if not commits:
        return 0, 0
	
    frequency = len(commits)
    developers = set()
    for commit in commits:
        developer = (commit.author.name.strip(), commit.author.email.strip())
        is_exist = any(is_same_developer(developer, existing_developer) for existing_developer in developers)
        if not is_exist:
            developers.add(developer)
            
    author_count = len(developers)

    return frequency, author_count


def calculate_file_frequency(raw_data_path, output_dir, repo_base_dir):
    """
     SATD “”：
    """
    # Read raw SATD data
    with open(raw_data_path, 'r', encoding='utf-8') as f:
        satd_data = json.load(f)

    features = []

    for entry in satd_data:
        print('-' * 60)
        satd_id = entry.get('satd_id')
        project_name = entry.get('project_name', '').strip()
        file_rel = entry.get('f_path', '').lstrip('/')
        add_commit_hash = entry.get('add_commit_hash', '').strip()

        # Default features
        feat = {
            'satd_id': satd_id,
            # 'developer_ownership': 0.0,
            'project_file_frequency': 0,
            'project_file_authors': 0,
        }

        # Repository path
        repo_path = os.path.join(repo_base_dir, project_name)
        repo = git.Repo(repo_path)

        # Calculate file stability: all commits for this file before (including) add_commit_hash
        frequency, author_count = get_file_frequency(repo, add_commit_hash, file_rel)

        feat['project_file_frequency'] = frequency
        feat['project_file_authors'] = author_count
        # feat['developer_ownership'] = round(frequency / author_count, 2) if author_count > 0 else 0.0
        features.append(feat)

        print(f"SATD {satd_id}@{add_commit_hash}: commits={frequency}, authors={author_count}")

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

    calculate_file_frequency(RAW_DATA_PATH, FEATURES_DIR, REPO_BASE_DIR)
