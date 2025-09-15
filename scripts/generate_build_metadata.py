#!/usr/bin/env python3
"""
Generate build metadata for reproducibility tracking.
"""

import os
import json
import hashlib
import subprocess
import datetime
from pathlib import Path

def get_git_info():
    """Get current git commit and status."""
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], 
                                       text=True, stderr=subprocess.DEVNULL).strip()
        return {
            'commit': commit,
            'dirty': subprocess.call(['git', 'diff', '--quiet'], 
                                   stderr=subprocess.DEVNULL) != 0
        }
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {'commit': 'unknown', 'dirty': False}

def get_python_info():
    """Get Python version and package info."""
    import sys
    import pkg_resources
    
    packages = []
    for dist in pkg_resources.working_set:
        packages.append({
            'name': dist.project_name,
            'version': dist.version
        })
    
    return {
        'python_version': sys.version,
        'packages': packages
    }

def calculate_file_hash(filepath):
    """Calculate SHA-1 hash of file."""
    if not os.path.exists(filepath):
        return None
    
    with open(filepath, 'rb') as f:
        return hashlib.sha1(f.read()).hexdigest()

def main():
    """Generate build metadata."""
    print("📊 Generating build metadata...")
    
    # Create build directory if it doesn't exist
    os.makedirs('build', exist_ok=True)
    
    # Collect metadata
    metadata = {
        'timestamp': datetime.datetime.now().isoformat(),
        'git': get_git_info(),
        'python': get_python_info(),
        'generated_files': {}
    }
    
    # Track generated files
    generated_patterns = [
        'tables_figures/latex/*.tex',
        'tables_figures/final_figures/*.pdf',
        'build/*.parquet'
    ]
    
    for pattern in generated_patterns:
        for filepath in Path('.').glob(pattern):
            if filepath.is_file():
                rel_path = str(filepath)
                metadata['generated_files'][rel_path] = {
                    'size': filepath.stat().st_size,
                    'sha1': calculate_file_hash(rel_path),
                    'mtime': datetime.datetime.fromtimestamp(filepath.stat().st_mtime).isoformat()
                }
    
    # Write metadata
    output_path = 'build/_RUNINFO.json'
    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Build metadata written to: {output_path}")
    print(f"   Generated files tracked: {len(metadata['generated_files'])}")
    print(f"   Git commit: {metadata['git']['commit'][:8]}")
    print(f"   Python version: {metadata['python']['python_version'].split()[0]}")

if __name__ == "__main__":
    main()
