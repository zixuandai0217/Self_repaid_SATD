#!/bin/bash
cd "$(dirname "$0")" || exit 1

echo "Geting raw_data.json from satd..."

python get_raw_data.py || { echo "Failed geting raw_data.json..."; exit 1; }