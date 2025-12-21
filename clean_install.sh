#!/bin/bash
# Clean installation script for Linux/macOS

echo "Cleaning old installation artifacts..."

# Remove egg-info and build directories
rm -rf sparse_linear_attention.egg-info
rm -rf build
rm -rf dist
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null

# Uninstall old package
echo "Uninstalling old package..."
pip uninstall -y sparse-linear-attention

# Reinstall with triton
echo "Installing with triton support..."
pip install -e .[triton]

echo ""
echo "Installation complete!"
