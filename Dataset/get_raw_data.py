import mysql.connector
import json
import os
from datetime import datetime

config = {
    'user': 'admin',          
    'password': '123456',  
    'host': 'localhost',     
    'database': 'satd'       
}

# Connect to database
cnx = mysql.connector.connect(**config)
cursor = cnx.cursor(dictionary=True)  

# Read query statement from external SQL file
sql_file_path = os.path.join(os.getcwd(), 'sql', 'query_self-fixed_satd.sql')  
with open(sql_file_path, 'r', encoding='utf-8') as f:
    query = f.read()

# Execute query
cursor.execute(query)

# Get all results
results = cursor.fetchall()

# Close cursor and database connection
cursor.close()
cnx.close()

# Use dictionary to record the earliest record corresponding to each satd_id
dedup_map = {}
for rec in results:
    satd_id = rec.get('satd_id')
    date = rec.get('add_date')
    if satd_id not in dedup_map or date < dedup_map[satd_id][0]:
        dedup_map[satd_id] = (date, rec)

# Extract deduplicated rec
deduped = [item[1] for item in dedup_map.values()]

# Get current directory
current_dir = os.getcwd()
file_path = os.path.join(current_dir, 'data', f'raw_data_{len(deduped)}.json')

# Write results to JSON file
with open(file_path, 'w', encoding='utf-8') as json_file:
    json.dump(deduped, json_file, indent=4, default=str) 


# Filter initial data, retain records with word count ≥ 2 in f_comment field
def count_words(s: str) -> int:
    return len(s.strip().split())

# 1. Read raw data
with open(file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 2. Filter out records with word count ≥ 2 in f_comment field
filtered = [rec for rec in data if count_words(rec.get("f_comment", "")) >= 2]

filtered_path = f"data/raw_data_filtered_{len(filtered)}.json"

# 3. Write filtered records
with open(filtered_path, 'w', encoding='utf-8') as f:
    json.dump(filtered, f, ensure_ascii=False, indent=2)

# 4. Read data after previous filtering
with open(filtered_path, 'r', encoding='utf-8') as f:
    filtered_data = json.load(f)

# 5. Further filtering: retain records with add_date ≤ remove_date
final_filtered = []
for rec in filtered_data:
    add_date = rec.get("add_date")
    remove_date = rec.get("remove_date")
    
    # Convert to datetime object for comparison
    d_add = datetime.fromisoformat(add_date)
    d_remove = datetime.fromisoformat(remove_date)
    # Only retain when add_date ≤ remove_date
    if d_add <= d_remove:
        final_filtered.append(rec)

# 6. Write final filtered records
final_path = os.path.join(current_dir, 'data', f'raw_data_final_{len(final_filtered)}.json')
with open(final_path, 'w', encoding='utf-8') as f:
    json.dump(final_filtered, f, ensure_ascii=False, indent=2)

# Print final statistics
print(f"Original record number: {len(deduped)}")
print(f"Retained (word count ≥ 2): {len(filtered)}")
print(f"Retained (add_date ≤ remove_date): {len(final_filtered)}, written to {final_path}")