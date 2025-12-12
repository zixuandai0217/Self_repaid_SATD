import os
import sys
import json
from datetime import datetime

def is_weekend_or_night(add_date_str):
    """
    If 996 (weekend or night), return 1, otherwise return 0.
    """
    dt = datetime.strptime(add_date_str, "%Y-%m-%d %H:%M:%S")

    # Weekend check: weekday() returns 0=Monday ... 6=Sunday
    if dt.weekday() >= 5:
        return 1

    # Night check: 22:00–23:59 or 00:00–05:59
    if dt.hour >= 21 or dt.hour < 9:
        return 1

    return 0

def process_add_date_flag(input_path, output_dir):
    with open(input_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    features = []

    for record in raw_data:
        print("-" * 60)
        satd_id = record.get("satd_id")
        add_date = record.get("add_date")
        
        flag_996 = is_weekend_or_night(add_date)
    
        features.append({
            "satd_id": satd_id,
            "satd_add_is_weekend_or_night": flag_996
        })
        print(f"satd_id: {satd_id} \nsatd_add_date: {add_date} \nsatd_add_is_weekend_or_night: {flag_996}")

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
