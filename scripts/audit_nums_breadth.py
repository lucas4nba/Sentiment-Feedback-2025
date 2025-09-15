#!/usr/bin/env python3
"""
Audit script to verify macro usage vs definition in breadth analysis.
"""

import os
import re
import sys
from pathlib import Path

def extract_macro_usage(filepath: str) -> set:
    """Extract macro usage from LaTeX file."""
    if not os.path.exists(filepath):
        return set()
    
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # Find all macro usages like \nbLowBOne, \nbVIXTripTwelve, etc.
    macro_pattern = r'\\nb[A-Za-z]+'
    matches = re.findall(macro_pattern, content)
    
    return set(matches)

def extract_macro_definitions(filepath: str) -> set:
    """Extract macro definitions from nums_breadth.tex."""
    if not os.path.exists(filepath):
        return set()
    
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # Find all macro definitions like \newcommand{\nbLowBOne}{...}
    macro_pattern = r'\\newcommand\{\\([^}]+)\}'
    matches = re.findall(macro_pattern, content)
    
    return set(f"\\{match}" for match in matches)

def main():
    """Main audit function."""
    print("🔍 Auditing breadth macro usage vs definition...")
    
    # Files to check
    main_tex = "main.tex"
    nums_breadth = "tables_figures/latex/nums_breadth.tex"
    
    # Extract usage and definitions
    usage = extract_macro_usage(main_tex)
    definitions = extract_macro_definitions(nums_breadth)
    
    print(f"\n📊 Summary:")
    print(f"  Macros used in prose: {len(usage)}")
    print(f"  Macros defined: {len(definitions)}")
    
    # Check for missing definitions
    missing = usage - definitions
    unused = definitions - usage
    
    if missing:
        print(f"\n❌ MISSING DEFINITIONS ({len(missing)}):")
        for macro in sorted(missing):
            print(f"  {macro}")
    
    if unused:
        print(f"\n⚠️  UNUSED DEFINITIONS ({len(unused)}):")
        for macro in sorted(unused):
            print(f"  {macro}")
    
    if not missing and not unused:
        print(f"\n✅ All macros properly defined and used!")
    
    # Exit with error code if missing macros
    if missing:
        print(f"\n❌ Audit failed: {len(missing)} missing macro definitions")
        return 1
    
    print(f"\n✅ Audit passed!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
