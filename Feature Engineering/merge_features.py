import json
import os
import argparse
from collections import defaultdict

# Define basic fields to retain
BASE_FIELDS = ["satd_id", "is_self_fixed", "project_name", ]

def load_base_data(raw_data_path):
    with open(raw_data_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    # Create basic data dictionary, with satd_id as key
    base_data_dict = {entry["satd_id"]: {field: entry.get(field) for field in BASE_FIELDS} for entry in raw_data}
    return base_data_dict

def merge_features(base_data_dict, feature_file):
    with open(feature_file, 'r', encoding='utf-8') as f:
        feature_data = json.load(f)
    
    for feature_entry in feature_data:
        if "satd_id" in feature_entry:
            # Use satd_id as primary key
            satd_id = feature_entry["satd_id"]
            if satd_id in base_data_dict:
                # Only add fields that don't exist, to avoid overwriting existing data
                for key, value in feature_entry.items():
                    if key != "satd_id" and key not in base_data_dict[satd_id]:
                        base_data_dict[satd_id][key] = value


def process_raw_data(raw_data_path, features_dir):
    base_data_dict = load_base_data(raw_data_path)

    feature_files = [f for f in os.listdir(features_dir) if f.endswith('.json')]
    
    # Merge specified feature files
    for feature_file in feature_files:
        feature_path = os.path.join(features_dir, feature_file)
        if os.path.exists(feature_path):
            merge_features(base_data_dict, feature_path)
            print(f"Merged feature file: {feature_file}")
    
    final_data = list(base_data_dict.values())
    
    # Write to merged_data.json file
    output_path = os.path.join(os.path.dirname(raw_data_path), f"merged_data_{len(final_data)}.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, indent=4)
    
    print(f"merged_data.json file generated: {output_path}")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    RAW_DATA_PATH = os.path.normpath(
        os.path.join(current_dir, "../Dataset/data/raw_data_final_40501.json")
    )
    FEATURES_DIR = os.path.normpath(
        os.path.join(current_dir, "features/")
    )

    process_raw_data(RAW_DATA_PATH, FEATURES_DIR)