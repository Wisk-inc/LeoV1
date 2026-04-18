#!/bin/bash
echo "Installing NEXUS VIDEO dependencies..."

# Python deps
pip install -r requirements.txt

# Create folders (already created by Jules but good for completeness)
mkdir -p models/wan22/{t2v,i2v,ti2v}
mkdir -p models/{qwen,rife,basicvsr}
mkdir -p outputs/{clips,merged,final}
mkdir -p temp

echo ""
echo "✅ NEXUS VIDEO ready!"
echo "Run: python app.py"
echo "Open: http://localhost:7860"
