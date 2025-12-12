import os
import json
import git

def calculate_file_features(raw_data_path, output_dir, repo_base_dir):
    """
    Calculate file line count, import module count, and SATD position ratio at the time of SATD introduction.
    """
    with open(raw_data_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    features = []

    for entry in raw_data:
        print("-" * 60)
        satd_id = entry.get('satd_id')
        project_name = entry.get('project_name')
        file_rel = entry.get('f_path', '').lstrip('/')
        add_commit_hash = entry.get('add_commit_hash', '').strip()
        start_line = entry.get('start_line', 0)

        feat = {
            'satd_id': satd_id,
            'code_file_lines': 0,
            'code_imported_modules': 0,
            'satd_position_in_file': 0.0,
        }

        repo_path = os.path.join(repo_base_dir, project_name)
        repo = git.Repo(repo_path)
        
        try:
            commit = repo.commit(add_commit_hash)
        except Exception as e:
            print(f"Unable to find commit {add_commit_hash} in {project_name}")
            features.append(feat)
            continue

        try:
            blob = commit.tree[file_rel]
            content = blob.data_stream.read().decode('utf-8')
        except KeyError:
            print(f"File {file_rel} does not exist in commit {add_commit_hash}")
            features.append(feat)
            continue
        except Exception as e:
            print(f"Error reading file content: {e}")
            features.append(feat)
            continue

        lines = content.splitlines()
        code_file_lines = len(lines)

        code_imported_modules = 0
        if file_rel.endswith('.java'):
            code_imported_modules = sum(1 for line in lines if line.strip().startswith('import '))
       
        satd_position = start_line / code_file_lines if code_file_lines > 0 else 0.0

        feat.update({
            'code_file_lines': code_file_lines,
            'code_imported_modules': code_imported_modules,
            'satd_position_in_file': satd_position
        })

        features.append(feat)
        print(f"SATD {satd_id}: lines={code_file_lines}, imports={code_imported_modules}, position={satd_position:.4f}")


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

    calculate_file_features(RAW_DATA_PATH, FEATURES_DIR, REPO_BASE_DIR)