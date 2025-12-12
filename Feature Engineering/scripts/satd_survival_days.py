import os
import sys
import json
from datetime import datetime
import math

def process_add_date_flag(input_path, output_dir):
    with open(input_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    features = []

    for record in raw_data:
        print("-" * 60)
        satd_id = record.get("satd_id")
        add_date = record.get("add_date")
        remove_date = record.get("remove_date")
        
        add_date_ts = datetime.strptime(add_date, "%Y-%m-%d %H:%M:%S").timestamp()
        remove_date_ts = datetime.strptime(remove_date, "%Y-%m-%d %H:%M:%S").timestamp()
        
        time_diff_seconds = remove_date_ts - add_date_ts
        
        survival_days = math.ceil(time_diff_seconds / (60*60*24)) if time_diff_seconds >= 0 else 0
    
        features.append({
            "satd_id": satd_id,
            "satd_survival_days": survival_days
        })
        print(f"satd_id: {satd_id} \nsatd_add_date: {add_date} \satd_survival_days: {survival_days}")

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

    process_add_date_flag(RAW_DATA_PATH, FEATURES_DIR)