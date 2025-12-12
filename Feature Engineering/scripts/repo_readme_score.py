from openai import OpenAI
import json
import os
import re
import sys
import subprocess

# Set API environment variables and LLM client
os.environ["OPENAI_API_KEY"] = ""
os.environ["OPENAI_API_BASE"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

client = OpenAI(
    base_url=os.environ["OPENAI_API_BASE"],
    api_key=os.environ["OPENAI_API_KEY"]
)

# Prompt and model invocation
def analyze_readme_score(readme_context: str) -> int:
    if not readme_context or readme_context.strip() == "":
        return 0
    """
    Score the content of README.md using the LLM API
    """
    dimension_judge_prompt_body = (
        "Score the content of README.md based on the following rubric, and only output the total score in the format <score>, nothing else."
        "\nScoring Rubric (Total: 5 points):"
        "\n1. [Content Completeness] (1 point) - Contains all essential sections: Project Overview, Installation, Usage, Contribution Guide, License"
        "\n2. [Structural Clarity] (1 point) - Has Table of Contents, proper heading hierarchy, logical flow, and working links"
        "\n3. [Richness of Examples] (1 point) - Includes code examples, screenshots/demos, and configuration examples"
        "\n4. [Maintenance Status] (1 point) - Shows recent update date, version badge, and community feedback link"
        "\n5. [Language Quality] (1 point) - Uses fluent language, clean formatting, and has no TODO/FIXME markers"
    )
    dimension_judge_prompt_examples = '''
Example 1:
README:
```md
# Project Overview
This tool automates data processing tasks.
# Installation
```bash
git clone ...
cd project
pip install -r requirements.txt
```
# Usage
```python
from tool import Processor
Processor.run('input.csv')
```
# Contribution
Submit PRs with tests.
# License
Apache-2.0
# FAQ
**Q**: How to set environment variables?
**A**: Use `export VAR=value`.
Last updated: 2025-05-03
![version badge](https://img.shields.io/badge/v1.0.0-blue.svg)
```
Response:
<5>

Example 2:
README:
```md
# Project Overview
A lightweight utility for file encryption.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Contribution](#contribution)
- [License](#license)

## Installation
```bash
npm install encrypt-tool
```

## Usage
```javascript
const encrypt = require('encrypt-tool');
encrypt.file('secret.txt', 'password123');
```

## Contribution
Fork the repo and submit pull requests.

## License
MIT License

Last updated: 2024-03-15  
![version badge](https://img.shields.io/npm/v/encrypt-tool)  
Join our [Slack channel](https://example.com/slack) for support
```
Response:
<4>

Example 3:
README:
```md
# MyLibrary
## Installation
```bash
pip install mylibrary
```
## Usage
```python
import mylibrary
mylibrary.do_work()
```
```
Response:
<3>

Example 4:
README:
```md
# Toolkit
## Installation
Download release from GitHub.
## Usage
No code examples provided.
Version: 2.1.4
```
Response:
<2>

Example 5:
README:
```md
# DemoProject
## Table of Contents
- [Install](#install)
- [Run](#run)

## Install
> TODO: add install steps
```
Response:
<1>

Example 6:
README:
```md
# SampleApp
Brief description missing sections.
```
Response:
<0>

'''
    prompt = dimension_judge_prompt_body + dimension_judge_prompt_examples + f"\nREADME:\n{readme_context}\nResponse:"

    # Call LLM API
    response = client.chat.completions.create(
        model="deepseek-v3", # Model Name
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    score_text = response.choices[0].message.content.strip()
    return extract(score_text)


def extract(response: str) -> int:
    matches = re.findall(r'<(\d+)>', response)
    return int(matches[-1]) if matches else 0

# (commit,file) -> score
readme_score_cache = {}

def get_readme_score(repo_path, start_commit_hash):
    try:
            
        # Get commit history (from current commit to earliest)
        rev_list = subprocess.run(
            ['git', '-C', repo_path, 'rev-list', start_commit_hash], check=True, capture_output=True, text=True, encoding='utf-8', errors='ignore'
        )

        commits = rev_list.stdout.splitlines() if rev_list.stdout else []

        for commit in commits:
            try:
                        
                # List all files in this commit
                ls_result = subprocess.run(
                    ['git', '-C', repo_path, 'ls-tree', '-r', '--name-only', commit], check=True, capture_output=True, text=True, encoding='utf-8', errors='ignore'
                )
                files = ls_result.stdout.splitlines() if ls_result.stdout else []
    
                # Check if there are readme-type files
                for file in files:
                    if 'readme' in file.lower():
                        print(f"Found README-type file: {file} ")
                        cache_key = (commit, file)

                        # Check if the score for this file is already in cache
                        if cache_key in readme_score_cache:
                            print(f"[✔ ] Score: {readme_score_cache[cache_key]} for {cache_key}")
                            return readme_score_cache[cache_key]

                        # Get file content
                        content_result = subprocess.run(
                            ['git', '-C', repo_path, 'show', f'{commit}:{file}'], check=True, capture_output=True, text=True, encoding='utf-8', errors='ignore'
                        )
                        readme_text = content_result.stdout.strip()

                        print(f"Read README content: {readme_text[:500]}...")  # Print first 500 characters

                        current_score = analyze_readme_score(readme_text) if readme_text else 0
                        
                        if current_score not in range(6):
                            print(f"[✖ ]Score {current_score} out of range, set to 0")
                            current_score = 0

                        # Store in cache
                        readme_score_cache[cache_key] = current_score

                        # Return score
                        return current_score

            except subprocess.CalledProcessError as e:
                print(f"Git command failed for commit {commit}: {e}")
                continue
                
    except subprocess.CalledProcessError as e:
        print(f"Git command failed for repo {repo_path}: {e}")
        return 0

def process_raw_data(input_path, output_dir, repo_base_dir):
    with open(input_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    features = []

    for record in raw_data:
        print('-' * 60)
        score = 0
        satd_id = record["satd_id"]
        project_name = record["project_name"]
        commit_hash = record["add_commit_hash"].strip()
        
        repo_path = os.path.join(repo_base_dir, project_name)
        
        result = get_readme_score(repo_path, commit_hash)

        score = result if result is not None else 0

        features.append({
            'satd_id': satd_id,
            'project_name': project_name,
            'project_readme_score': score
        })
        print(f"Processed SATD ID: {satd_id}, Project: {project_name}, README Score: {score}")

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