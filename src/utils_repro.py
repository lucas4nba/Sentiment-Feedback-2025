import os
import random
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Union, List, Dict, Any

import numpy as np
import pandas as pd

# Try to import gitpython, but don't fail if not available
try:
    import git
    GIT_AVAILABLE = True
except ImportError:
    GIT_AVAILABLE = False


def set_global_seed(seed: Union[int, None]) -> int:
    """
    Set global random seeds for numpy and python random.
    
    Args:
        seed: Integer seed or None to read from SENTFEED_SEED env var (default: 123456)
    
    Returns:
        The seed value that was used
    """
    if seed is None:
        seed = int(os.environ.get('SENTFEED_SEED', 123456))
    
    # Set numpy random seed
    np.random.seed(seed)
    
    # Set python random seed
    random.seed(seed)
    
    # Also set the default numpy random generator
    np.random.default_rng(seed)
    
    return seed


def sample_hash(df: pd.DataFrame, id_cols: List[str]) -> str:
    """
    Create a hash of the sorted unique ID rows.
    
    Args:
        df: DataFrame to hash
        id_cols: List of column names to use as IDs
    
    Returns:
        First 10 characters of SHA1 hash
    """
    # Get unique rows based on ID columns
    unique_rows = df[id_cols].drop_duplicates().sort_values(id_cols)
    
    # Convert to string representation and hash
    row_str = unique_rows.to_string(index=False)
    hash_obj = hashlib.sha1(row_str.encode('utf-8'))
    
    return hash_obj.hexdigest()[:10]


def write_runinfo(tag: str, info: Dict[str, Any]) -> None:
    """
    Write run information to JSON files for reproducibility tracking.
    
    Args:
        tag: Tag for this analysis run
        info: Dictionary of additional information to log
    """
    # Get current timestamp
    timestamp = datetime.now().isoformat()
    
    # Get Python version
    python_version = sys.version
    
    # Try to get git commit info
    git_commit = None
    if GIT_AVAILABLE:
        try:
            repo = git.Repo(search_parent_directories=True)
            git_commit = repo.head.object.hexsha[:8]  # First 8 chars
        except (git.InvalidGitRepositoryError, git.GitCommandError):
            pass
    
    # Prepare run info
    run_info = {
        'timestamp': timestamp,
        'tag': tag,
        'git_commit': git_commit,
        'python_version': python_version,
        'info': info
    }
    
    # Create logs directory structure
    logs_dir = Path('logs/analysis_runs')
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Write to JSONL file
    jsonl_path = logs_dir / f'{tag}_runs.jsonl'
    with open(jsonl_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(run_info) + '\n')
    
    # Create build directory and write latest run info
    build_dir = Path('build')
    build_dir.mkdir(exist_ok=True)
    
    runinfo_path = build_dir / '_RUNINFO.json'
    with open(runinfo_path, 'w', encoding='utf-8') as f:
        json.dump(run_info, f, indent=2)


# Export the three main functions
__all__ = ['set_global_seed', 'sample_hash', 'write_runinfo']
