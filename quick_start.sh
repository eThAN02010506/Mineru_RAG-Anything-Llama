
#!/bin/bash
# Quick start script for RAG-Anything on macOS M3

# Text colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "========================================="
echo "RAG-Anything Quick Start for macOS M3"
echo "========================================="

# Check if running on macOS
if [[ "$(uname)" != "Darwin" ]]; then
    echo "Warning: This script is designed for macOS. Some features may not work on your system."
fi

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo -e "\
Creating Python virtual environment..."
    python3 -m venv .venv
    if [ $? -ne 0 ]; then
        echo "Failed to create virtual environment. Please install Python 3.8+ and try again."
        exit 1
    fi
    echo "Virtual environment created successfully."
else
    echo -e "\
Virtual environment already exists."
fi

# Activate virtual environment
echo -e "\
Activating virtual environment..."
source .venv/bin/activate
if [ $? -ne 0 ]; then
    echo "Failed to activate virtual environment."
    exit 1
fi
echo "Virtual environment activated."

# Check if dependencies are already installed
echo -e "\
Checking Python dependencies..."
MISSING_DEPS=0
python -c "import sentence_transformers" 2>/dev/null || MISSING_DEPS=1
python -c "import faiss" 2>/dev/null || MISSING_DEPS=1
python -c "import numpy" 2>/dev/null || MISSING_DEPS=1
python -c "import PyPDF2" 2>/dev/null || MISSING_DEPS=1

# Only install dependencies if they're missing and we have internet
if [ $MISSING_DEPS -eq 1 ]; then
    echo "Some dependencies are missing. Checking internet connection..."
    
    # Check for internet connection
    if ping -c 1 google.com &> /dev/null; then
        echo "Internet connection available. Installing dependencies..."
        pip install -r requirements.txt
        if [ $? -ne 0 ]; then
            echo "Some dependencies may have failed to install. Continuing anyway..."
        else
            echo "Dependencies installed successfully."
        fi
    else
        echo "No internet connection detected. Skipping dependency installation."
        echo "Warning: Some required packages may be missing. The system may not work correctly."
    fi
else
    echo "All required dependencies are already installed."
fi

# Check for fix_chunks_issue.py and copy it to the current directory if needed
if [ ! -f "fix_chunks_issue.py" ]; then
    echo -e "\
Setting up fix_chunks_issue.py..."
    # Check if it exists in the workspace directory
    if [ -f "/workspace/fix_chunks_issue.py" ]; then
        cp /workspace/fix_chunks_issue.py .
        echo "Copied fix_chunks_issue.py to the current directory."
    else
        echo "Warning: fix_chunks_issue.py not found. Some repair functions may not work."
    fi
fi

# Install Poppler for PDF processing only if not already installed
echo -e "\
Checking for Poppler (pdftotext)..."
if ! command -v pdftotext &> /dev/null; then
    echo "pdftotext not found. Checking internet connection for installation..."
    
    # Check for internet connection
    if ping -c 1 google.com &> /dev/null; then
        echo "Internet connection available. Attempting to install Poppler..."
        if command -v brew &> /dev/null; then
            brew install poppler
            if [ $? -ne 0 ]; then
                echo "Failed to install Poppler with Homebrew. PDF processing may be limited."
            else
                echo "Poppler installed successfully."
            fi
        else
            echo "Homebrew not found. Please install Poppler manually for better PDF processing."
        fi
    else
        echo "No internet connection detected. Skipping Poppler installation."
        echo "Warning: PDF processing capabilities may be limited without pdftotext."
    fi
else
    echo "Poppler is already installed."
fi

# Create data directory if it doesn't exist
if [ ! -d "data" ]; then
    echo -e "\
Creating data directory..."
    mkdir -p data
    echo "Data directory created."
else
    echo -e "\
Data directory already exists."
fi

# Check for PDF files in data directory
pdf_count=$(find data -name "*.pdf" | wc -l)
if [ $pdf_count -eq 0 ]; then
    echo -e "\
No PDF files found in the data directory."
    echo "Please add PDF files to the 'data' directory before running RAG-Anything."
else
    echo -e "\
Found $pdf_count PDF file(s) in the data directory."
fi

# Check for embedding model
if [ ! -d "models/all-MiniLM-L6-v2" ]; then
    echo -e "\
Embedding model not found. You need to download it manually."
    echo "Please download the all-MiniLM-L6-v2 model and place it in the models directory."
else
    echo -e "\
Embedding model found."
fi

# Check for language model only if not already present
if [ ! -f "models/llama/mistral-7b.gguf" ]; then
    echo -e "\
Language model not found. Checking internet connection..."
    
    # Check for internet connection
    if ping -c 1 google.com &> /dev/null; then
        echo "Internet connection available. Attempting to download..."
        python download_model.py
        if [ $? -ne 0 ]; then
            echo "Failed to download language model. Please download it manually."
        else
            echo "Language model downloaded successfully."
        fi
    else
        echo "No internet connection detected. Skipping model download."
        echo "Warning: Language model is required for RAG-Anything to work."
    fi
else
    echo -e "\
Language model found."
fi

# Run system check
echo -e "\
Running system check..."
python check_system.py

# Final instructions
echo -e "\
========================================="
echo "Setup complete!"
echo -e "\
To use RAG-Anything:"
echo "1. Make sure your PDF files are in the 'data' directory"
echo "2. Run: python run_rag.py ./data"
echo "3. Or ask a specific question: python run_rag.py ./data \"What is this document about?\""
echo -e "\
If you encounter any issues:"
echo "1. Try fixing: python run_rag.py ./data --fix"
echo "2. Try reprocessing: python run_rag.py ./data --reparse"
echo "3. Check the README.md file for more troubleshooting tips"
echo -e "\
========================================="
