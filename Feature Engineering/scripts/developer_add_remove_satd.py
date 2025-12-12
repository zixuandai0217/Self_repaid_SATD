import os
import json
import mysql.connector
from datetime import datetime
import sys

# Database connection configuration (please modify according to actual situation)
DB_CONFIG = {
    'user': 'admin',
    'password': '123456',
    'host': 'localhost',
    'database': 'satd',
    'charset': 'utf8mb4'
}

def calculate_developer_history(raw_json_path, output_dir):
    with open(raw_json_path, 'r', encoding='utf-8') as f:
        entries = json.load(f)

    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor(dictionary=True)
    
    features = []

    for entry in entries:
        print('-' * 60)
        satd_id = entry.get('satd_id')
        project_name = entry.get('project_name')
        adder_email = entry.get('adder_email', '').strip()
        adder_name = entry.get('adder_name', '').strip()
        add_date = entry['add_date']

        # Query total Added and Removed counts
        sql_counts = '''
        SELECT
            SUM(CASE WHEN s.resolution = 'SATD_ADDED' THEN 1 ELSE 0 END) AS added_count,
            SUM(CASE WHEN s.resolution = 'SATD_REMOVED' THEN 1 ELSE 0 END) AS removed_count
        FROM satd.SATD s
        JOIN satd.Projects p ON s.p_id = p.p_id
        LEFT JOIN satd.Commits c ON s.p_id = c.p_id AND s.second_commit = c.commit_hash
        WHERE p.p_name = %s AND (c.author_name = %s OR c.author_email = %s) AND c.author_date < %s;
        '''
        cursor.execute(sql_counts, (project_name, adder_name, adder_email, add_date))
        row = cursor.fetchone() or {}
        added = int(row.get('added_count') or 0)
        removed = int(row.get('removed_count') or 0)

        features.append({
            'satd_id': satd_id,
            'project_name': project_name,
            'developer_added_satd_count': added,
            'developer_removed_satd_count': removed,
        })
        print(f"SATD {satd_id}: Author:{adder_name}@{adder_email} Added={added}, Removed={removed}")

    script_name = os.path.splitext(os.path.basename(__file__))[0]
    output_path = os.path.join(output_dir, f"{script_name}.json")
    
    os.makedirs(output_dir, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(features, f, indent=4, ensure_ascii=False)
    print(f"Feature file successfully generated: {output_path}")

    cursor.close()
    conn.close()


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    RAW_DATA_PATH = os.path.normpath(
        os.path.join(current_dir, "../../Dataset/data/raw_data_final_40501.json")
    )
    FEATURES_DIR = os.path.normpath(
        os.path.join(current_dir, "../features/")
    )

    calculate_developer_history(RAW_DATA_PATH, FEATURES_DIR)
