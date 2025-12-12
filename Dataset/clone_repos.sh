#!/bin/bash

# Create repos directory (if not exists)
mkdir -p repos

# Read each line of the csv file
while IFS= read -r repo; do
    # Extract organization name and project name
    org=$(echo "$repo" | awk -F'/' '{print $4}')
    project=$(echo "$repo" | awk -F'/' '{print $5}')

    # Create folder with organization name
    mkdir -p "repos/$org"

    # Clone project to the corresponding folder
    git clone "$repo" "repos/$org/$project"

done < repos.csv
