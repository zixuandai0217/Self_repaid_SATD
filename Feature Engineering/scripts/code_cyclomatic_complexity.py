import json
import os
import re
import sys

def calculate_cyclomatic_complexity(code):
    """Accurately calculate the cyclomatic complexity of Java method body"""
    if not code or code.strip() == "":
        return 0  
    
    complexity = 0

    # Decision point detection (includes Java main control flow structures)
    decision_patterns = [
        r'\bif\s*\(',                # if statement
        r'\belse\s+if\s*\(',         # else if statement
        r'\bfor\s*\(',               # for loop
        r'\bwhile\s*\(',             # while loop
        r'\bcase\s+',                # switch-case statement
        r'\bdefault\s*:',            # default case
        r'\|\|',                     # logical OR
        r'&&',                       # logical AND
        r'\bcatch\s*\(',             # exception catch
        r'\bthrow\b',                # throw statement
        r'\?.*?:'                    # ternary operator
    ]
    
    # Remove comment content
    code = re.sub(r'//.*|/\*.*?\*/', '', code, flags=re.DOTALL)
    
    # Split code lines for analysis
    for line in code.split('\n'):
        line = line.strip()
        # Skip empty lines and documentation comments
        if not line or line.startswith('*'):
            continue  
            
        for pattern in decision_patterns:
            matches = list(re.finditer(pattern, line))
            complexity += len(matches)

    return complexity

def process_raw_data(input_path, output_dir):
    with open(input_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    features = []
    for record in raw_data:
        print('-' * 60)
        satd_id = record.get("satd_id")
        method_body = record.get('method_body', '').strip()
        
        complexity = calculate_cyclomatic_complexity(method_body)
        
        features.append({
            "satd_id": satd_id,
            "code_cyclomatic_complexity": complexity
        })
        print(f"Processed SATD ID: {satd_id}, code_cyclomatic_complexity: {complexity}")

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
    
    process_raw_data(RAW_DATA_PATH, FEATURES_DIR)
