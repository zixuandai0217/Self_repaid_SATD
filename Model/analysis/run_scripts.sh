#!/bin/bash
cd "$(dirname "$0")" || exit 1

# 1. Execute all feature extraction scripts
for script in *.py; do
    echo "------------------------"
    echo "📊 Running: $script......"
    python "$script" || { echo "❌ Failed: $script......"; exit 1; }
	echo "✅ Success: $script......"
	echo "------------------------"
done

echo "🎉 All done!!!"