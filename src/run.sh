#!/bin/bash
# Run Expert Jasmine pipeline for STAI-X 2026

echo "=================================="
echo "STAI-X 2026 - Expert Jasmine"
echo "=================================="
echo ""

# Check if dependencies are installed
echo "Checking dependencies..."
python3 -c "import lightgbm, numpy, pandas, PIL, sklearn" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Missing dependencies. Installing..."
    pip install -r requirements.txt
else
    echo "✅ All dependencies installed"
fi

echo ""
echo "Running pipeline..."
echo ""

# Run main script
cd src
python3 main.py

echo ""
echo "=================================="
echo "Pipeline complete!"
echo "=================================="
