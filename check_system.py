
#!/usr/bin/env python3
"""
System check script for RAG-Anything on macOS M3
Verifies that all components are working correctly
"""
import os
import sys
import subprocess
import importlib
import platform

def check_python_version():
    """Check Python version"""
    print("Checking Python version...")
    version = sys.version_info
    print(f"   Python version: {version.major}.{version.minor}.{version.micro}")
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("ERROR: Python version must be at least 3.8")
        return False
    print("SUCCESS: Python version is sufficient")
    return True

def check_platform():
    """Check platform information"""
    print("Checking platform information...")
    system = platform.system()
    machine = platform.machine()
    print(f"   System: {system}")
    print(f"   Machine: {machine}")
    
    if system != "Darwin":
        print("WARNING: This script is optimized for macOS, but may work on other platforms")
    
    if "arm64" in machine.lower() or "aarch64" in machine.lower():
        print("SUCCESS: Running on ARM architecture (Apple Silicon)")
    else:
        print("WARNING: Not running on ARM architecture, performance may be affected")
    
    return True

def check_dependencies():
    """Check if required Python packages are installed"""
    print("Checking Python dependencies...")
    required_packages = [
        "sentence_transformers",
        "faiss",
        "numpy",
        "PyPDF2"
    ]
    
    optional_packages = [
        "llama_cpp",
        "textract"
    ]
    
    all_required_installed = True
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"SUCCESS: {package} is installed")
        except ImportError:
            print(f"ERROR: {package} is NOT installed")
            all_required_installed = False
    
    for package in optional_packages:
        try:
            importlib.import_module(package)
            print(f"SUCCESS: {package} is installed")
        except ImportError:
            print(f"WARNING: Optional package {package} is NOT installed")
    
    return all_required_installed

def check_external_tools():
    """Check if required external tools are installed"""
    print("Checking external tools...")
    
    # Check pdftotext
    try:
        subprocess.run(["pdftotext", "-v"], check=True, capture_output=True)
        print("SUCCESS: pdftotext is installed")
        pdftotext_installed = True
    except (subprocess.SubprocessError, FileNotFoundError):
        print("WARNING: pdftotext is NOT installed")
        pdftotext_installed = False
    
    # If pdftotext is not installed, suggest installation
    if not pdftotext_installed:
        print("   To install pdftotext:")
        print("   brew install poppler")
    
    return True

def check_models():
    """Check if required models are available"""
    print("Checking models...")
    
    # Check embedding model
    script_dir = os.path.dirname(os.path.abspath(__file__))
    embedding_model_path = os.path.join(script_dir, "models", "all-MiniLM-L6-v2")
    
    if os.path.exists(embedding_model_path):
        print(f"SUCCESS: Embedding model found at: {embedding_model_path}")
    else:
        print(f"ERROR: Embedding model NOT found at: {embedding_model_path}")
        print("   You need to download the embedding model.")
    
    # Check LLM
    llm_path = os.path.join(script_dir, "models", "llama", "mistral-7b.gguf")
    
    if os.path.exists(llm_path):
        print(f"SUCCESS: Language model found at: {llm_path}")
    else:
        print(f"WARNING: Language model NOT found at: {llm_path}")
        print("   You can download it using: python download_model.py")
    
    return True

def check_files():
    """Check if required script files are available"""
    print("Checking script files...")
    
    required_files = [
        "run_rag.py",
        "parse_with_mineru.py",
        "embed.py",
        "query.py"
    ]
    
    all_files_present = True
    for file in required_files:
        if os.path.exists(file):
            print(f"SUCCESS: {file} is present")
        else:
            print(f"ERROR: {file} is MISSING")
            all_files_present = False
    
    return all_files_present

def check_data_directory():
    """Check if data directory is set up correctly"""
    print("Checking data directory...")
    
    data_dir = "./data"
    if not os.path.exists(data_dir):
        print(f"WARNING: Data directory not found: {data_dir}")
        print("   Creating data directory...")
        os.makedirs(data_dir, exist_ok=True)
    
    pdf_files = [f for f in os.listdir(data_dir) if f.lower().endswith('.pdf')]
    if pdf_files:
        print(f"SUCCESS: Found {len(pdf_files)} PDF files in data directory")
    else:
        print(f"WARNING: No PDF files found in data directory")
        print("   Please add PDF files to the data directory")
    
    return True

def main():
    """Main function to run all checks"""
    print("=" * 60)
    print("RAG-Anything System Check for macOS M3")
    print("=" * 60)
    
    checks = [
        ("Python Version", check_python_version),
        ("Platform Information", check_platform),
        ("Python Dependencies", check_dependencies),
        ("External Tools", check_external_tools),
        ("Models", check_models),
        ("Script Files", check_files),
        ("Data Directory", check_data_directory)
    ]
    
    all_passed = True
    for name, check_func in checks:
        print("\
" + "-" * 60)
        print(f"Checking {name}...")
        try:
            result = check_func()
            if not result:
                all_passed = False
        except Exception as e:
            print(f"ERROR during {name} check: {e}")
            all_passed = False
    
    print("\
" + "=" * 60)
    if all_passed:
        print("SUCCESS: All critical checks passed!")
        print("   You should be able to run RAG-Anything successfully.")
        print("   Try: python run_rag.py ./data")
    else:
        print("WARNING: Some checks failed. Please address the issues above.")
    print("=" * 60)

if __name__ == "__main__":
    main()
