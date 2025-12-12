#!/bin/bash
cd "$(dirname "$0")" || exit 1


for script in *.py; do
    echo "------------------------"
    echo "📊 Running: $script......"
    python "$script" || { echo "❌ Failed: $script......"; exit 1; }
	echo "✅ Success: $script......"
	echo "------------------------"
done


for script in *.R; do
    echo "------------------------"
    echo "📊 Running: $script......"
    Rscript "$script" || { echo "❌ Failed: $script......"; exit 1; }
	echo "✅ Success: $script......"
	echo "------------------------"
done

echo "🎉 All done!!!"