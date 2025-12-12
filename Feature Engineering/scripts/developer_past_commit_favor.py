from tkinter.tix import MAX
from openai import OpenAI
import json
import os
import re
import subprocess
from collections import Counter
import time

# Set API environment variables and OpenAI client
os.environ["OPENAI_API_KEY"] = ""
os.environ["OPENAI_API_BASE"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

client = OpenAI(
    base_url=os.environ["OPENAI_API_BASE"],
    api_key=os.environ["OPENAI_API_KEY"]
)

# Prompt and model invocation
def analyze_developer_past_commit_favor(commit_hash: str, commit_message: str) -> str:
    if not commit_message or commit_message.strip() == "":
        return "other"
    
    if commit_hash in commit_cache:
        print("[✔ ] Retrieved result from cache")
        return commit_cache[commit_hash]
    
    """
    Use LLM interface to classify comments and return classification results.
    """
    dimension_judge_prompt_body = (
        "You are an expert code-review assistant."
		"Your task is to read a single git commit message and decide **its primary purpose**."
        "Choose exactly one tag from the list below:"
		"1. <feature>   — Adds new functionality or user-visible feature\n"
		"2. <bugfix>    — Fixes a bug, error, or incorrect behavior\n"
		"3. <refactor>  — Refactors or restructures code without changing behavior\n"
		"4. <cleanup>   — Removes dead code, comments, or performs non-functional cleanup\n"  
		"5. <other>     — Anything else (build scripts, config, dependency bumps, etc.)"
        "\nOutput only the tag.only the tag."
        "\nAnalyze carefully."
        "\n\n"
    )
    dimension_judge_prompt_examples = '''
Example1:
Commit Message:  
`Add OAuth2 login flow with Google provider`  
Response: `<feature>`

Commit Message:  
`Fix null-pointer exception in UserService when email is missing`  
Response: `<bugfix>`

Commit Message:  
`Refactor OrderController to use service layer abstraction`  
Response: `<refactor>`

Commit Message:  
`Remove unused helper methods and obsolete classes`  
Response: `<cleanup>`

Commit Message:  
`Bump dependency versions and warnings`  
Response: `<other>`

'''
    prompt = dimension_judge_prompt_body + dimension_judge_prompt_examples + f"\nCommit Message:\n{commit_message}\nResponse:"

    # Call OpenAI API
    try:
        response = client.chat.completions.create(
            model="deepseek-v3",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        score = extract(response.choices[0].message.content.strip())
        
    except Exception as e:
        print(f"Error analyzing commit {commit_hash}: {e}")
        time.sleep(60)
        score = "other"

    print(f"commit_hash: {commit_hash}, message: {commit_message}, score: {score}")

    # Update cache
    commit_cache[commit_hash] = score
    
    return score

def extract(response: str) -> str:
    allowed_types = {'feature', 'bugfix', 'refactor', 'cleanup', 'docs', 'perf', 'test', 'other'}
    matches = re.findall(r'<([^>]+)>', response)
    if matches:
        last_type = matches[-1].strip()
        return last_type if last_type in allowed_types else 'other'
    return 'other'


# Default maximum number of commits
MAX_COMMITS = 10

commit_cache = {}

def get_commits(repo_dir, author_name, author_email, add_date) -> list:
    cmd = [
            "git", "-C", repo_dir, "log",
            f"--author={author_name}", # Match either name or email
            f"--author={author_email}",
            f"-n{MAX_COMMITS}",
            f"--before={add_date}",
            "--pretty=format:%H|%s" # Use commit hash and commit message
    ]
    try:
        output = subprocess.run(cmd, check=True, capture_output=True, text=True, encoding='utf-8', errors='ignore')
        
        commits = [] # hash: message
        for line in output.stdout.splitlines():
            if "|" in line: 
                commit_hash, commit_message = line.split("|", 1)
                commits.append({
                    "commit_hash": commit_hash.strip(),
                    "commit_message": commit_message.strip()
                })
        return commits
    
    except subprocess.CalledProcessError:
        return []


def process_raw_data(input_path, output_dir, repo_base_dir):
    with open(input_path, "r", encoding="utf-8") as f:
        raw_records = json.load(f)

    features = []

    for rec in raw_records:
        print("-" * 60)
        satd_id = rec.get("satd_id")
        project_name = rec.get("project_name")
        name = rec.get("adder_name")
        email = rec.get("adder_email")
        add_date = rec.get("add_date")
        
        repo_path = os.path.join(repo_base_dir, project_name)
        
		# Get the developer's previous n commits
        commits = get_commits(repo_path, name, email, add_date)
        
        total = len(commits)
        
        print(f"Processing {name}@{email} --> Total Commits: {total}")
        
        counts = Counter()
        for commit in commits:
            commit_hash = commit["commit_hash"]
            commit_message = commit["commit_message"]
            category = analyze_developer_past_commit_favor(commit_hash, commit_message)
            counts[category] += 1
            print(f"Commit: {commit_message} -> Category: {category}")

        bugfix_ratio = counts['bugfix'] / total if total else 0.0 # Past bug fix ratio
        cleanup_ratio = counts['cleanup'] / total if total else 0.0 # Past cleanup ratio
        feature_ratio = counts['feature'] / total if total else 0.0 # Past feature addition ratio
        refactor_ratio = counts['refactor'] / total if total else 0.0 # Past refactoring ratio


        features.append({
			"satd_id": satd_id,
			"developer_past_bugfix_ratio": bugfix_ratio,
			"developer_past_cleanup_ratio": cleanup_ratio,
            "developer_past_feature_ratio": feature_ratio,
            "developer_past_refactor_ratio": refactor_ratio,
		})
        
        print(f"Processing record {satd_id}:\n{bugfix_ratio}\n{cleanup_ratio}\n{feature_ratio}\n{refactor_ratio}")
        time.sleep(1)
        

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
    