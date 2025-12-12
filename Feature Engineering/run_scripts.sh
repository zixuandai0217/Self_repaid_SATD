#!/bin/bash
cd "$(dirname "$0")" || exit 1

# 1. 
for script in scripts/*.py; do
    echo "Running: $script"
    python "$script" || { echo "Failed: $script..."; exit 1; }
done

# 2. 
echo "Merging features..."
python merge_feature.py || { echo "Merge failed..."; exit 1; }

echo "All done!!!  Final features: ../Dataset/data/merged_data.json"