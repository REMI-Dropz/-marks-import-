#!/bin/bash
echo "Starting custom build script..."
echo "Python version:"
python --version
echo "Pip version:"
pip --version
echo "Upgrading pip..."
pip install --upgrade pip
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt --no-cache-dir --index-url https://pypi.python.org/simple
if [ $? -eq 0 ]; then
    echo "Dependencies installed successfully."
    echo "Listing installed packages..."
    pip list
else
    echo "Failed to install dependencies."
    exit 1
fi
echo "Build script completed."