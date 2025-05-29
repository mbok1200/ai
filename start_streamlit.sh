#!/bin/bash

$VIRTUAL_ENV="venv"
# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 is not installed."
    exit 1
fi
# Check if inside a virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    echo "❌ Not inside a virtual environment."
    source $VIRTUAL_ENV/bin/activate
else
    echo "✅ Inside a virtual environment: $VIRTUAL_ENV"
fi
# Check for .env
if [ -f ".env" ]; then
  echo ".env exists ✅"
else
  echo ".env is missing ❌"
  exit 1
fi

# Check for credentials.json
if [ -f "config/credentials.json" ]; then
  echo "credentials.json exists ✅"
else
  echo "credentials.json is missing ❌"
  exit 1
fi
# Check if requirements.txt exists
if [ ! -f "requirements.txt" ]; then
    echo "❌ requirements.txt not found."
    exit 1
fi

# Check if all requirements are installed
echo "🔍 Checking Python dependencies..."
missing=$(pip3 check 2>&1 | grep -v "No broken requirements found.")

if [ -z "$missing" ]; then
    echo "✅ All dependencies are satisfied."
else
    echo "⚠️ Some packages are missing or broken:"
    echo "$missing"
    echo "➡️ Running: pip install -r requirements.txt"
    pip install -r requirements.txt || { echo "❌ Failed to install requirements."; exit 1; }
fi
echo "Google drive getting started"
# check if the user has gdrive file map json
if [ -f "config/gdrive_file_map.json" ]; then
  echo "gdrive_file_map.json exists ✅"
else
  echo "gdrive_file_map.json is missing ❌"
  python3 GoogleDrive.py
fi

echo "Starting Streamlit app"
streamlit run config/talk_to_your_file_v3.py --server.runOnSave=false
