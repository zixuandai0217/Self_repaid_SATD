import os
import sys
import json
import subprocess


files_count_cache = {}
# key (commit_hash) -> file_count

def get_java_files(repo_path, commit_hash):
    """Get the total number of Java files at the specified commit state"""
    if commit_hash in files_count_cache:
        print(f"[✔ ] {commit_hash} -> {files_count_cache[commit_hash]}")
        return files_count_cache[commit_hash]
    try:
        # List all file paths under the specified commit
        ls_cmd = ['git', '-C', repo_path, 'ls-tree', '-r', '--name-only', commit_hash]
        ls_result = subprocess.run(ls_cmd, check=True, capture_output=True, text=True, encoding='utf-8', errors='ignore')
        
        files = ls_result.stdout.splitlines() if ls_result.stdout else []
        java_count = sum(1 for f in files if f.lower().endswith('.java'))
    except subprocess.CalledProcessError as e:
        java_count = 0

    # Write result to cache and return
    files_count_cache[commit_hash] = java_count
    print(f"[✘ ] {commit_hash} -> {java_count}")
    return java_count

def analyze_project_files(input_path, output_dir, repo_base_dir):
    with open(input_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    features = []
    for record in raw_data:
        print("-" * 60)
        satd_id = record["satd_id"]
        project_name = record["project_name"]
        commit_hash = record["add_commit_hash"]

        # Construct repository path
        repo_path = os.path.join(repo_base_dir, project_name)
        
        # Get Java file count
        count_files = get_java_files(repo_path, commit_hash)

        features.append({
            "satd_id": satd_id,
            "project_files": count_files
        })
        print(f"Project {project_name}@{commit_hash} Java file count: {count_files}")

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
        os.path.join(current_dir, "../../Dataset/repos/")
    )

    analyze_project_files(RAW_DATA_PATH, FEATURES_DIR, REPO_BASE_DIR)
