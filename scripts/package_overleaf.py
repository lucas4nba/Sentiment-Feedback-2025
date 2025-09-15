"""
Overleaf packaging script for sentiment analysis paper.

Creates a complete Overleaf-ready archive with all necessary LaTeX files,
figures, and tables. Automatically detects and includes all referenced files.
"""

import os
import zipfile
import argparse
import logging
import re
from pathlib import Path
from typing import Set, List, Dict
from datetime import datetime


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def find_latex_files(root_dir: str) -> Set[str]:
    """
    Find all LaTeX files in the project.
    
    Parameters
    ----------
    root_dir : str
        Root directory to search
        
    Returns
    -------
    Set[str]
        Set of LaTeX file paths
    """
    logger = logging.getLogger(__name__)
    
    latex_files = set()
    
    # Common LaTeX extensions
    latex_extensions = {'.tex', '.bib', '.cls', '.sty'}
    
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if any(file.endswith(ext) for ext in latex_extensions):
                file_path = os.path.join(root, file)
                latex_files.add(file_path)
                logger.debug(f"Found LaTeX file: {file_path}")
    
    logger.info(f"Found {len(latex_files)} LaTeX files")
    return latex_files


def find_input_references(content: str) -> Set[str]:
    """
    Find all \\input{} references in LaTeX content.
    
    Parameters
    ----------
    content : str
        LaTeX file content
        
    Returns
    -------
    Set[str]
        Set of referenced file paths
    """
    # Pattern to match \input{...} commands
    input_pattern = r'\\input\{([^}]+)\}'
    matches = re.findall(input_pattern, content)
    
    # Also check for \IfFileExists{...}{\input{...}}
    iffile_pattern = r'\\IfFileExists\{([^}]+)\}'
    iffile_matches = re.findall(iffile_pattern, content)
    
    all_refs = set(matches + iffile_matches)
    return all_refs


def find_includegraphics_references(content: str) -> Set[str]:
    """
    Find all \\includegraphics{} references in LaTeX content.
    
    Parameters
    ----------
    content : str
        LaTeX file content
        
    Returns
    -------
    Set[str]
        Set of referenced image paths
    """
    # Pattern to match \includegraphics[...]{...} commands
    graphics_pattern = r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}'
    matches = re.findall(graphics_pattern, content)
    
    # Also check for \IfFileExists{...}{\includegraphics{...}}
    iffile_graphics_pattern = r'\\IfFileExists\{([^}]+)\}.*?\\includegraphics'
    iffile_matches = re.findall(iffile_graphics_pattern, content, re.DOTALL)
    
    all_refs = set(matches + iffile_matches)
    return all_refs


def resolve_file_path(ref_path: str, base_dir: str) -> str:
    """
    Resolve a referenced file path relative to base directory.
    
    Parameters
    ----------
    ref_path : str
        Referenced file path
    base_dir : str
        Base directory for resolution
        
    Returns
    -------
    str
        Resolved file path
    """
    # Remove leading/trailing whitespace
    ref_path = ref_path.strip()
    
    # If path is already absolute, return as-is
    if os.path.isabs(ref_path):
        return ref_path
    
    # Try different possible locations
    possible_paths = [
        os.path.join(base_dir, ref_path),
        os.path.join(base_dir, 'tables_figures', ref_path),
        os.path.join(base_dir, 'tables_figures', 'latex', ref_path),
        os.path.join(base_dir, 'tables_figures', 'final_figures', ref_path),
        os.path.join(base_dir, 'FIGS', ref_path),
        os.path.join(base_dir, 'FIGS', 'robustness', ref_path)
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    # Return the original path if not found
    return ref_path


def collect_referenced_files(latex_files: Set[str], root_dir: str) -> Set[str]:
    """
    Collect all files referenced by LaTeX files.
    
    Parameters
    ----------
    latex_files : Set[str]
        Set of LaTeX file paths
    root_dir : str
        Root directory
        
    Returns
    -------
    Set[str]
        Set of all referenced file paths
    """
    logger = logging.getLogger(__name__)
    
    referenced_files = set()
    
    for latex_file in latex_files:
        try:
            with open(latex_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Find \input{} references
            input_refs = find_input_references(content)
            for ref in input_refs:
                resolved_path = resolve_file_path(ref, root_dir)
                if os.path.exists(resolved_path):
                    referenced_files.add(resolved_path)
                    logger.debug(f"Found input reference: {ref} -> {resolved_path}")
                else:
                    logger.warning(f"Input reference not found: {ref}")
            
            # Find \includegraphics{} references
            graphics_refs = find_includegraphics_references(content)
            for ref in graphics_refs:
                resolved_path = resolve_file_path(ref, root_dir)
                if os.path.exists(resolved_path):
                    referenced_files.add(resolved_path)
                    logger.debug(f"Found graphics reference: {ref} -> {resolved_path}")
                else:
                    logger.warning(f"Graphics reference not found: {ref}")
                    
        except Exception as e:
            logger.error(f"Error processing {latex_file}: {e}")
    
    logger.info(f"Found {len(referenced_files)} referenced files")
    return referenced_files


def create_overleaf_archive(root_dir: str, output_path: str) -> None:
    """
    Create Overleaf-ready archive.
    
    Parameters
    ----------
    root_dir : str
        Root directory of the project
    output_path : str
        Output ZIP file path
    """
    logger = logging.getLogger(__name__)
    
    # Find all LaTeX files
    latex_files = find_latex_files(root_dir)
    
    # Collect all referenced files
    referenced_files = collect_referenced_files(latex_files, root_dir)
    
    # Combine all files to include
    all_files = latex_files | referenced_files
    
    # Create manifest
    manifest = {
        'created': datetime.now().isoformat(),
        'total_files': len(all_files),
        'latex_files': len(latex_files),
        'referenced_files': len(referenced_files),
        'files': sorted(list(all_files))
    }
    
    logger.info(f"Creating Overleaf archive with {len(all_files)} files")
    
    # Create ZIP archive
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Add all files
        for file_path in all_files:
            # Calculate relative path from root
            rel_path = os.path.relpath(file_path, root_dir)
            
            # Normalize path separators for cross-platform compatibility
            rel_path = rel_path.replace('\\', '/')
            
            zipf.write(file_path, rel_path)
            logger.debug(f"Added to archive: {rel_path}")
        
        # Add manifest
        import json
        manifest_content = json.dumps(manifest, indent=2)
        zipf.writestr('_OVERLEAF_MANIFEST.json', manifest_content)
    
    logger.info(f"Overleaf archive created: {output_path}")
    
    # Print manifest summary
    print(f"\nOVERLEAF ARCHIVE CREATED: {output_path}")
    print(f"Total files: {manifest['total_files']}")
    print(f"LaTeX files: {manifest['latex_files']}")
    print(f"Referenced files: {manifest['referenced_files']}")
    print(f"Manifest saved as: _OVERLEAF_MANIFEST.json")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Create Overleaf-ready archive')
    parser.add_argument('--root-dir', type=str, default='.',
                       help='Root directory of the project (default: current directory)')
    parser.add_argument('--output', type=str, default='SENTIMENT_OVERLEAF.zip',
                       help='Output ZIP file path (default: SENTIMENT_OVERLEAF.zip)')
    parser.add_argument('--output-dir', type=str, default='/mnt/data',
                       help='Output directory (default: /mnt/data)')
    
    args = parser.parse_args()
    
    logger = setup_logging()
    logger.info("Starting Overleaf packaging")
    
    try:
        # Resolve output path
        if args.output_dir:
            output_path = os.path.join(args.output_dir, args.output)
        else:
            output_path = args.output
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Create archive
        create_overleaf_archive(args.root_dir, output_path)
        
        logger.info("Overleaf packaging completed successfully")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise


if __name__ == "__main__":
    main()
