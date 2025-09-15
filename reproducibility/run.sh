#!/bin/bash
# One-button orchestrator for Sentiment Feedback analysis reproduction
# Implements the 5-step process with JSON run logging

set -euo pipefail  # Exit on error, undefined vars, pipe failures

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
RUN_LOG_FILE="$SCRIPT_DIR/RUN_LOG.json"
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Default options
DRY_RUN=false
VERBOSE=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --verbose|-v)
            VERBOSE=true
            shift
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            show_help
            exit 1
            ;;
    esac
done

# Help function
show_help() {
    cat << EOF
Sentiment Feedback Analysis Reproduction Script

Usage: $0 [options]

Options:
    --dry-run              Show commands without executing
    --verbose, -v          Enable verbose output
    --help, -h             Show this help message

This script implements a 5-step reproduction process:
1. Environment check
2. Make environment validation
3. Data preparation
4. Analysis execution (IRF, GMM, panel, portfolios, figures, tables)
5. JSON run log generation

EOF
}

# Logging functions
log_section() {
    echo -e "\n${PURPLE}========================================${NC}"
    echo -e "${PURPLE}$1${NC}"
    echo -e "${PURPLE}========================================${NC}"
}

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Command execution wrapper
run_cmd() {
    local cmd="$1"
    local description="${2:-Running command}"
    
    if [[ "$VERBOSE" == true ]]; then
        log_info "$description: $cmd"
    else
        log_info "$description"
    fi
    
    if [[ "$DRY_RUN" == true ]]; then
        echo "  [DRY RUN] $cmd"
        return 0
    fi
    
    if ! eval "$cmd"; then
        log_error "Command failed: $cmd"
        exit 1
    fi
}

# Get git commit hash
get_git_commit() {
    if command -v git >/dev/null 2>&1 && [[ -d ".git" ]]; then
        git rev-parse HEAD 2>/dev/null || echo "unknown"
    else
        echo "unknown"
    fi
}

# Write JSON run log
write_run_log() {
    local status="$1"
    local start_time="$2"
    local end_time="$3"
    local git_commit="$4"
    
    cat > "$RUN_LOG_FILE" << EOF
{
  "run_id": "$TIMESTAMP",
  "status": "$status",
  "start_time": "$start_time",
  "end_time": "$end_time",
  "duration_seconds": $(($(date -d "$end_time" +%s) - $(date -d "$start_time" +%s))),
  "git_commit": "$git_commit",
  "steps_completed": [
    "environment_check",
    "make_env",
    "make_data", 
    "make_analysis",
    "run_log_generation"
  ],
  "environment": {
    "python_version": "$(python3 --version 2>&1)",
    "platform": "$(uname -s)",
    "architecture": "$(uname -m)"
  },
  "outputs": {
    "figures": "project_files/tables_figures/final_figures/",
    "tables": "project_files/tables_figures/latex/",
    "data": "project_files/build/"
  }
}
EOF
}

# Main reproduction pipeline
main() {
    local start_time="$TIMESTAMP"
    local git_commit=$(get_git_commit)
    
    log_section "Sentiment Feedback Analysis Reproduction"
    log_info "Start time: $start_time"
    log_info "Git commit: $git_commit"
    log_info "Project directory: $PROJECT_DIR"
    
    # Change to project directory
    cd "$PROJECT_DIR"
    
    # Step 1: Environment check
    log_section "Step 1/5: Environment Check"
    run_cmd "python reproducibility/check_env.py" "Running comprehensive environment check"
    
    # Step 2: Make environment validation
    log_section "Step 2/5: Make Environment Validation"
    run_cmd "make -C reproducibility env" "Validating environment with Make"
    
    # Step 3: Data preparation
    log_section "Step 3/5: Data Preparation"
    run_cmd "make -C reproducibility data" "Building all data files"
    
    # Step 4: Analysis execution
    log_section "Step 4/5: Analysis Execution"
    log_info "Running: IRF estimation, GMM calibration, panel analysis, portfolios, figures, tables"
    run_cmd "make -C reproducibility irf gmm panel ports figures tables" "Executing complete analysis pipeline"
    
    # Step 5: Success - write run log
    log_section "Step 5/5: Run Log Generation"
    local end_time=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    run_cmd "write_run_log 'success' '$start_time' '$end_time' '$git_commit'" "Writing JSON run log"
    
    log_section "Reproduction Complete"
    log_success "All steps completed successfully!"
    log_info "Run log saved to: $RUN_LOG_FILE"
    
    # Summary
    log_info "Outputs generated:"
    echo "  📊 Figures: project_files/tables_figures/final_figures/"
    echo "  📋 Tables: project_files/tables_figures/latex/"
    echo "  💾 Data: project_files/build/"
    echo "  📝 Log: $RUN_LOG_FILE"
    
    log_success "Reproduction pipeline finished successfully!"
}

# Error handling
trap 'log_error "Script failed at line $LINENO. Check the output above for details."; exit 1' ERR

# Check if we're in the right directory
if [[ ! -f "project_files/README.md" ]]; then
    log_error "Please run this script from the project root directory"
    exit 1
fi

# Run main function
main "$@"