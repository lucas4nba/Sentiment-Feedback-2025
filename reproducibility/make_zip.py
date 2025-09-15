#!/usr/bin/env python3
"""
Helper script for creating reproducibility packages.
Reads manifest.yaml and validates all expected outputs exist before zipping.
"""

import os
import sys
import yaml
import json
from pathlib import Path
from typing import Dict, List, Any

def load_manifest(manifest_path: str) -> Dict[str, Any]:
    """Load and parse manifest.yaml file."""
    try:
        with open(manifest_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"❌ ERROR: Manifest file not found: {manifest_path}")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"❌ ERROR: Invalid YAML in manifest: {e}")
        sys.exit(1)

def validate_outputs(artifact_dir: str, manifest: Dict[str, Any]) -> bool:
    """
    Validate that all expected outputs exist in the artifact directory.
    
    Args:
        artifact_dir: Path to artifact directory
        manifest: Parsed manifest.yaml content
    
    Returns:
        True if all outputs exist, False otherwise
    """
    artifact_path = Path(artifact_dir)
    
    print("Validating expected outputs...")
    print("-" * 50)
    
    all_valid = True
    
    # Check analysis outputs
    if 'analysis_outputs' in manifest:
        for output_id, output_info in manifest['analysis_outputs'].items():
            if 'produces' in output_info:
                for output_file in output_info['produces']:
                    file_path = artifact_path / output_file
                    if file_path.exists():
                        print(f"✅ {output_file}")
                    else:
                        print(f"❌ {output_file} - MISSING")
                        all_valid = False
    
    # Check figures
    if 'figures' in manifest:
        for fig_id, fig_info in manifest['figures'].items():
            if 'path' in fig_info:
                # Extract filename from path
                fig_path = Path(fig_info['path'])
                file_path = artifact_path / fig_path.name
                if file_path.exists():
                    print(f"✅ {fig_path.name}")
                else:
                    print(f"❌ {fig_path.name} - MISSING")
                    all_valid = False
    
    # Check tables
    if 'tables' in manifest:
        for table_id, table_info in manifest['tables'].items():
            if 'path' in table_info:
                # Extract filename from path
                table_path = Path(table_info['path'])
                file_path = artifact_path / table_path.name
                if file_path.exists():
                    print(f"✅ {table_path.name}")
                else:
                    print(f"❌ {table_path.name} - MISSING")
                    all_valid = False
    
    # Check required documentation files
    required_docs = ['README.md', 'manifest.yaml']
    for doc_file in required_docs:
        file_path = artifact_path / doc_file
        if file_path.exists():
            print(f"✅ {doc_file}")
        else:
            print(f"❌ {doc_file} - MISSING")
            all_valid = False
    
    # Check optional files
    optional_files = ['RUN_LOG.json']
    for opt_file in optional_files:
        file_path = artifact_path / opt_file
        if file_path.exists():
            print(f"✅ {opt_file}")
        else:
            print(f"⚠️  {opt_file} - OPTIONAL (not found)")
    
    print("-" * 50)
    
    if all_valid:
        print("✅ All required outputs validated successfully")
    else:
        print("❌ Some required outputs are missing")
    
    return all_valid

def get_file_stats(artifact_dir: str) -> Dict[str, Any]:
    """Get statistics about files in the artifact directory."""
    artifact_path = Path(artifact_dir)
    
    stats = {
        'total_files': 0,
        'total_size_bytes': 0,
        'file_types': {},
        'directories': []
    }
    
    for item in artifact_path.rglob('*'):
        if item.is_file():
            stats['total_files'] += 1
            stats['total_size_bytes'] += item.stat().st_size
            
            # Count file types
            ext = item.suffix.lower()
            if ext:
                stats['file_types'][ext] = stats['file_types'].get(ext, 0) + 1
            else:
                stats['file_types']['no_extension'] = stats['file_types'].get('no_extension', 0) + 1
        elif item.is_dir():
            stats['directories'].append(str(item.relative_to(artifact_path)))
    
    return stats

def format_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"

def main():
    """Main function."""
    if len(sys.argv) != 2:
        print("Usage: python make_zip.py <artifact_directory>")
        print("Example: python make_zip.py reproducibility/artifacts/abc123")
        sys.exit(1)
    
    artifact_dir = sys.argv[1]
    
    if not os.path.exists(artifact_dir):
        print(f"❌ ERROR: Artifact directory not found: {artifact_dir}")
        sys.exit(1)
    
    print("=" * 60)
    print("Reproducibility Package Validator")
    print("=" * 60)
    print(f"Artifact directory: {artifact_dir}")
    print()
    
    # Load manifest
    manifest_path = os.path.join(artifact_dir, 'manifest.yaml')
    if not os.path.exists(manifest_path):
        print(f"❌ ERROR: Manifest file not found: {manifest_path}")
        print("Make sure manifest.yaml is in the artifact directory")
        sys.exit(1)
    
    manifest = load_manifest(manifest_path)
    print(f"✅ Manifest loaded: {manifest.get('metadata', {}).get('project', 'Unknown project')}")
    print()
    
    # Validate outputs
    validation_passed = validate_outputs(artifact_dir, manifest)
    print()
    
    # Get file statistics
    stats = get_file_stats(artifact_dir)
    print("Package Statistics:")
    print("-" * 30)
    print(f"Total files: {stats['total_files']}")
    print(f"Total size: {format_size(stats['total_size_bytes'])}")
    print(f"Directories: {len(stats['directories'])}")
    
    if stats['file_types']:
        print("\nFile types:")
        for ext, count in sorted(stats['file_types'].items()):
            print(f"  {ext}: {count} files")
    
    print()
    
    if validation_passed:
        print("✅ Package validation PASSED")
        print("Ready for distribution!")
        sys.exit(0)
    else:
        print("❌ Package validation FAILED")
        print("Please ensure all required outputs are present before packaging")
        sys.exit(1)

if __name__ == "__main__":
    main()
