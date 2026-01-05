#!/bin/bash
#
# Auto-Haaland Setup Script
#
# This script sets up the development environment for the Auto-Haaland project.
# It requires sudo access to install system packages.
#

set -e

echo "=========================================="
echo "Auto-Haaland Development Environment Setup"
echo "=========================================="
echo ""

# Check if running on Debian/Ubuntu
if [ -f /etc/debian_version ]; then
    echo "Detected Debian/Ubuntu system"
    echo ""

    echo "Step 1: Installing system packages..."
    sudo apt update
    sudo apt install -y python3.12-venv docker-compose
    echo "✓ System packages installed"
    echo ""
fi

echo "Step 2: Creating virtual environment..."
if [ -d "venv" ]; then
    echo "Removing existing virtual environment..."
    rm -rf venv
fi
python3 -m venv venv
echo "✓ Virtual environment created"
echo ""

echo "Step 3: Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"
echo ""

echo "Step 4: Upgrading pip..."
pip install --upgrade pip
echo "✓ pip upgraded"
echo ""

echo "Step 5: Installing Python dependencies..."
pip install -r requirements-dev.txt
echo "✓ Python dependencies installed"
echo ""

echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Activate the virtual environment:"
echo "     source venv/bin/activate"
echo ""
echo "  2. Start LocalStack:"
echo "     make local-up"
echo ""
echo "  3. Run tests:"
echo "     make test-unit"
echo ""
echo "See QUICKSTART.md for more details!"
