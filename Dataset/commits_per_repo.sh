#!/bin/bash

# Output file
output_file="commits_per_repo.csv"

# Write CSV header
echo "org,project,commits" > "$output_file"

# Loop through all repository directories
for org_dir in repos/*; do
    # Ensure it is a folder
    [ -d "$org_dir" ] || continue

    org=$(basename "$org_dir")

    for project_dir in "$org_dir"/*; do
        [ -d "$project_dir/.git" ] || continue  # Ensure it is a git repository

        project=$(basename "$project_dir")

        # Enter repository directory, count commits
        cd "$project_dir" || continue
        commits=$(git rev-list --count HEAD 2>/dev/null || echo 0)
        cd - > /dev/null || exit

        # Write to CSV
        echo "$org,$project,$commits" >> "$output_file"
        echo "✅ $org/$project: $commits commits"
    done
done

echo "🎯 Statistics completed, results saved to $output_file"
