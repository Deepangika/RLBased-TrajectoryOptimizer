#!/usr/bin/env python3
"""
Setup Gemini API and run CEM batch training.

Set your API key using one of these methods:
1. Set environment variable: $env:GOOGLE_API_KEY="your_key_here"
2. Pass as argument: python setup_and_run.py "your_key_here"
3. Enter interactively when prompted
"""
import sys
import os
import subprocess
from pathlib import Path

def get_api_key():
    """Get Gemini API key from various sources."""
    
    # Check if passed as argument
    if len(sys.argv) > 1:
        key = sys.argv[1]
        if key and not key.startswith("-"):
            return key
    
    # Check environment variables
    key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if key:
        print(f"✅ Found API key in environment")
        return key
    
    # Ask user
    print("\n" + "="*80)
    print("GEMINI API KEY REQUIRED")
    print("="*80)
    print("\nYour Gemini API key is needed to run training.")
    print("Get it from: https://aistudio.google.com/app/apikey\n")
    
    key = input("Enter your Gemini API key: ").strip()
    
    if not key:
        print("❌ No API key provided")
        return None
    
    return key

def main():
    """Setup and run batch training."""
    
    # Get API key
    api_key = get_api_key()
    if not api_key:
        print("Cannot proceed without API key")
        sys.exit(1)
    
    # Set environment variable
    os.environ["GOOGLE_API_KEY"] = api_key
    print(f"✅ API key configured")
    
    # Run batch training
    print("\n" + "="*80)
    print("Starting CEM batch training...")
    print("="*80 + "\n")
    
    project_root = Path(__file__).resolve().parent
    script = project_root / "scripts" / "batch_run_cem.py"
    
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=project_root,
        env={**os.environ, "GOOGLE_API_KEY": api_key}
    )
    
    sys.exit(result.returncode)

if __name__ == "__main__":
    main()
