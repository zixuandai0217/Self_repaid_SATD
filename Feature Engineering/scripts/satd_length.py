import os
import sys
import json

def analyze_comment_word_count(record):
    """Analyze the word count in the comments"""
    return len(record["f_comment"].strip().split())

def process_comment_word_count(input_path, output_dir):
    with open(input_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    features = []
    for record in raw_data:      
        word_count = analyze_comment_word_count(record)
        
        feature_record = {
            "satd_id": record["satd_id"],
            "satd_length": word_count
        }
        features.append(feature_record)

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
    
    process_comment_word_count(RAW_DATA_PATH, FEATURES_DIR)