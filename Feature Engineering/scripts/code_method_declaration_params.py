import json
import os
import sys

def get_method_parameters(method_declaration):
    if not method_declaration:
        return 0
    
    # Find the start and end positions of the parameter list
    start = method_declaration.find('(')
    end = method_declaration.rfind(')')
    
    # If parentheses not found, return 0
    if start == -1 or end == -1 or start >= end:
        return 0
    
    # Extract the parameter list part
    params_str = method_declaration[start+1:end].strip()
    
    # If parameter list is empty, return 0
    if not params_str:
        return 0
    
    # Handle complex parameter cases (generics, nested parentheses, etc.) e.g. public <T> T getFirst(List<T> list)
    depth = 0  # Parentheses nesting depth
    params = []  # Parameter list
    current_param = []  # Currently building parameter
    
    for char in params_str:
        if char in '<({[':
            depth += 1
            current_param.append(char)
        elif char in '>)}]':
            depth -= 1
            current_param.append(char)
        elif char == ',' and depth == 0:
            # Encounter top-level comma, complete a parameter
            param_str = ''.join(current_param).strip()
            if param_str:
                params.append(param_str)
            current_param = []
        else:
            current_param.append(char)
    
    # Add the last parameter
    if current_param:
        param_str = ''.join(current_param).strip()
        if param_str:
            params.append(param_str)
    
    return len(params)

def process_raw_data(input_path, output_dir):
    with open(input_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    features = []

    for record in raw_data:
        print("-" * 60)
        satd_id = record.get("satd_id")
        method_declaration = record.get("method_declaration")
        
        method_parameters = get_method_parameters(method_declaration)
        
        # Add result to feature list
        features.append({
            "satd_id": satd_id,
            "code_method_declaration_params": method_parameters
        })
        print(f"satd_id: {satd_id}, code_method_declaration_params: {method_parameters}")

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