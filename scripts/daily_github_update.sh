#!/bin/bash

# Daily GitHub Update Script for GNN_Trans
# Usage: ./daily_github_update.sh [commit_message]
#
# Repository policy:
# - Local repo keeps the full working state.
# - GitHub is a public subset of the local repo, not a full mirror.
# - Files inside OPEN_SOURCE_PATHS are the public surface.
# - Files inside OPEN_SOURCE_FORCE_PATHS are exact public exceptions to broad
#   local ignore rules; keep this list file-specific.
# - Files inside OPEN_SOURCE_EXCLUDE_PATHS are removed from the GitHub subset
#   even if they still exist locally.
# - Files matching OPEN_SOURCE_EXCLUDE_PATHSPECS are local artifacts and are
#   removed after whitelist staging.

set -e  # Exit on any error

# Configuration
PROJECT_DIR="/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans"
BRANCH_NAME="main"
DEFAULT_COMMIT_MSG="Daily update - $(date '+%Y-%m-%d %H:%M')"
OPEN_SOURCE_PATHS=(
    "src/"
    "configs/"
    "danger_gen_ablation/"
    ".gitignore"
    "__init__.py"
    "README.md"
    "archive/README.md"
    "data/README.md"
    "experiments/README.md"
    "outputs/README.md"
    "config.py"
    "advanced_analysis.py"
    "aggregate_result_type_logs.py"
    "analyze_distribution.py"
    "analyze_training_results.py"
    "generate_danger_scenes.py"
    "generate_medium_speed_manifest.py"
    "generate_stratified_manifest.py"
    "heatmap_only.py"
    "idm_predictor.py"
    "lstm_predictor.py"
    "process_by_scenario_type.py"
    "process_nuplan_to_12d.py"
    "run_gnn_manifest_eval.py"
    "run_idm_pkl_eval.py"
    "run_lstm_pkl_eval.py"
    "run_lstm_train_and_eval.sh"
    "run_medium_speed_3000_comparison.sh"
    "run_medium_speed_3000_from_manifest.sh"
    "run_nuboard_simulation.py"
    "run_transformer_only_manifest_eval.py"
    "scripts/daily_github_update.sh"
    "scripts/analysis/"
    "scripts/eval_closed_loop/"
    "scripts/preprocessing/"
    "scripts/training/check_split_consistency.py"
    "scripts/training/eval_test_set.py"
    "scripts/training/magnet_imitation_pooled.py"
    "scripts/training/perturbation.py"
    "scripts/training/prebuild_graphdataset_cache.py"
    "scripts/training/run_2x2_ablation.py"
    "scripts/training/split_log_disjoint.py"
    "scripts/training/split_log_disjoint3.py"
    "scripts/training/tests/test_gnn_pooling.py"
    "scripts/training/tests/test_perturbation.py"
    "scripts/training/train_magnet_imitation.py"
    "devkit_patches/"
    "train.py"
    "train_diffusion_generator.py"
    "train_diffusion_gnn_trans.py"
    "train_lstm.py"
    "train_transformer_only.py"
    "visualize_evaluation.py"
    "visualize_training_logs.py"
    "docs/README.md"
    "docs/TRAJECTORY_PREDICTION.md"
    "docs/DANGER_GENERATION.md"
    "docs/feature_schema_v2.md"
    "docs/tutorials/"
    "docs/reference/"
    "docs/reports/results_analysis_report.md"
    "docs/reports/summary.md"
)
OPEN_SOURCE_FORCE_PATHS=(
    # Exact legacy-code destinations retained as public Git renames. Do not
    # replace these entries with archive/, which contains large local data.
    "archive/legacy_pre_v8_20260610/advanced_analysis.py"
    "archive/legacy_pre_v8_20260610/aggregate_result_type_logs.py"
    "archive/legacy_pre_v8_20260610/analyze_distribution.py"
    "archive/legacy_pre_v8_20260610/analyze_training_results.py"
    "archive/legacy_pre_v8_20260610/generate_danger_scenes.py"
    "archive/legacy_pre_v8_20260610/heatmap_only.py"
    "archive/legacy_pre_v8_20260610/idm_predictor.py"
    "archive/legacy_pre_v8_20260610/lstm_predictor.py"
    "archive/legacy_pre_v8_20260610/run_gnn_manifest_eval.py"
    "archive/legacy_pre_v8_20260610/run_idm_pkl_eval.py"
    "archive/legacy_pre_v8_20260610/run_lstm_pkl_eval.py"
    "archive/legacy_pre_v8_20260610/run_lstm_train_and_eval.sh"
    "archive/legacy_pre_v8_20260610/run_medium_speed_3000_comparison.sh"
    "archive/legacy_pre_v8_20260610/run_medium_speed_3000_from_manifest.sh"
    "archive/legacy_pre_v8_20260610/run_nuboard_simulation.py"
    "archive/legacy_pre_v8_20260610/run_transformer_only_manifest_eval.py"
    "archive/legacy_pre_v8_20260610/scripts/analysis/advanced_analysis.py"
    "archive/legacy_pre_v8_20260610/scripts/analysis/aggregate_result_type_logs.py"
    "archive/legacy_pre_v8_20260610/scripts/analysis/analyze_distribution.py"
    "archive/legacy_pre_v8_20260610/scripts/analysis/analyze_training_results.py"
    "archive/legacy_pre_v8_20260610/scripts/analysis/heatmap_only.py"
    "archive/legacy_pre_v8_20260610/scripts/analysis/inspect_generated_traj.py"
    "archive/legacy_pre_v8_20260610/scripts/analysis/smoke_test_sdedit.py"
    "archive/legacy_pre_v8_20260610/scripts/analysis/visualize_evaluation.py"
    "archive/legacy_pre_v8_20260610/scripts/analysis/visualize_training_logs.py"
    "archive/legacy_pre_v8_20260610/src/danger_generation/core/dataset/danger_pair_dataset.py"
    "archive/legacy_pre_v8_20260610/train.py"
    "archive/legacy_pre_v8_20260610/train_diffusion_generator.py"
    "archive/legacy_pre_v8_20260610/train_lstm.py"
    "archive/legacy_pre_v8_20260610/train_transformer_only.py"
    "archive/legacy_pre_v8_20260610/visualize_evaluation.py"
    "archive/legacy_pre_v8_20260610/visualize_training_logs.py"

    # Scenario-set manifests are public provenance inputs, despite the broad
    # local *.txt ignore rule.
    "scripts/eval_closed_loop/logs_pittsburgh_heldout.txt"
    "scripts/eval_closed_loop/logs_pittsburgh_heldout_clean.txt"
    "scripts/eval_closed_loop/logs_pittsburgh_test60.txt"
    "scripts/eval_closed_loop/logs_pittsburgh_val60.txt"
)
OPEN_SOURCE_EXCLUDE_PATHS=(
    ".claude/"
    "docs/archive/"
    "docs/work_records/"
    "src/danger_generation/core/danger_pair_scenarios/"
    "src/ego_planning/nuboard/gnn_trans_planner(backup).py"
    "src/ego_planning/planner/_gnn_trans_planner_backup.py"
    "src/ego_planning/planner/gnn_trans_planner_backup(backup).py"
    "src/ego_planning/planner/gnn_transformer_planner_backup(backup).py"
    "danger_gen_ablation/__pycache__/"
    "danger_gen_ablation/results/"
)
OPEN_SOURCE_EXCLUDE_PATHSPECS=(
    # Backups are local recovery artifacts, never public source files.
    ":(glob)**/*.bak*"
)
OPEN_SOURCE_FORBIDDEN_SYMLINKS=(
    "advanced_analysis.py"
    "aggregate_result_type_logs.py"
    "analyze_distribution.py"
    "analyze_training_results.py"
    "generate_medium_speed_manifest.py"
    "generate_stratified_manifest.py"
    "heatmap_only.py"
    "process_by_scenario_type.py"
    "process_nuplan_to_12d.py"
    "visualize_evaluation.py"
    "visualize_training_logs.py"
)

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Print step message
print_step() {
    echo -e "${YELLOW}=== $1 ===${NC}"
}

# Print status message
print_status() {
    echo -e "${GREEN}$1${NC}"
}

# Print warning message
print_warning() {
    echo -e "${RED}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}错误: $1${NC}"
}

# Check if in git repository
check_git_repo() {
    if [ ! -d ".git" ]; then
        echo -e "${RED}错误: 当前目录不是Git仓库${NC}"
        exit 1
    fi
}

# Check for changes
check_changes() {
    git status --porcelain | grep -q '^[ AMRD]'
}

# Get commit message
get_commit_message() {
    if [ -n "$1" ]; then
        echo "$1"
    else
        echo "$DEFAULT_COMMIT_MSG"
    fi
}

extract_status_path() {
    local status_line="$1"
    local path_part

    path_part="${status_line:3}"
    if [[ "$path_part" == *" -> "* ]]; then
        path_part="${path_part##* -> }"
    fi

    printf '%s\n' "$path_part"
}

is_excluded_path() {
    local candidate_path="$1"
    local exclude_path

    for exclude_path in "${OPEN_SOURCE_EXCLUDE_PATHS[@]}"; do
        if [[ "$candidate_path" == "$exclude_path" ]] || [[ "$candidate_path" == "$exclude_path"* ]]; then
            return 0
        fi
    done

    return 1
}

collect_open_source_status_lines() {
    local status_line
    local candidate_path

    while IFS= read -r status_line; do
        [ -z "$status_line" ] && continue
        candidate_path=$(extract_status_path "$status_line")
        if is_excluded_path "$candidate_path"; then
            continue
        fi
        printf '%s\n' "$status_line"
    done < <(git status --short -- "${OPEN_SOURCE_PATHS[@]}")
}

create_temp_index() {
    local temp_index

    temp_index=$(mktemp)
    GIT_INDEX_FILE="$temp_index" git read-tree HEAD

    printf '%s\n' "$temp_index"
}

stage_open_source_paths() {
    local index_file="${1:-}"
    local path

    print_step "Staging open-source whitelist paths..."

    if [ -n "$index_file" ]; then
        for path in "${OPEN_SOURCE_PATHS[@]}"; do
            GIT_INDEX_FILE="$index_file" git add -- "$path" 2>/dev/null || true
        done
        for path in "${OPEN_SOURCE_FORCE_PATHS[@]}"; do
            GIT_INDEX_FILE="$index_file" git add -f -- "$path" 2>/dev/null || true
        done
        for path in "${OPEN_SOURCE_EXCLUDE_PATHS[@]}"; do
            GIT_INDEX_FILE="$index_file" git rm -r --cached --ignore-unmatch -q -- "$path" 2>/dev/null || true
        done
        for path in "${OPEN_SOURCE_EXCLUDE_PATHSPECS[@]}"; do
            GIT_INDEX_FILE="$index_file" git rm -r --cached --ignore-unmatch -q -- "$path" 2>/dev/null || true
        done
        return 0
    fi

    git reset -q >/dev/null
    for path in "${OPEN_SOURCE_PATHS[@]}"; do
        git add -- "$path" 2>/dev/null || true
    done
    for path in "${OPEN_SOURCE_FORCE_PATHS[@]}"; do
        git add -f -- "$path" 2>/dev/null || true
    done

    for path in "${OPEN_SOURCE_EXCLUDE_PATHS[@]}"; do
        git rm -r --cached --ignore-unmatch -q -- "$path" 2>/dev/null || true
    done
    for path in "${OPEN_SOURCE_EXCLUDE_PATHSPECS[@]}"; do
        git rm -r --cached --ignore-unmatch -q -- "$path" 2>/dev/null || true
    done
}

collect_staged_subset_diff() {
    local index_file="$1"

    GIT_INDEX_FILE="$index_file" git diff --cached --name-status
}

validate_no_forbidden_public_symlinks() {
    local path

    for path in "${OPEN_SOURCE_FORBIDDEN_SYMLINKS[@]}"; do
        if [ -L "$path" ]; then
            print_error "公开兼容入口 $path 仍是 symlink。请先改为普通 Python wrapper 文件后再上传。"
            exit 1
        fi
    done
}

preview_open_source_changes() {
    local temp_index
    local preview_output

    cd "$PROJECT_DIR"
    check_git_repo
    validate_no_forbidden_public_symlinks

    print_step "Previewing open-source whitelist changes..."
    temp_index=$(create_temp_index)
    trap 'rm -f "$temp_index"' EXIT
    stage_open_source_paths "$temp_index"
    preview_output=$(collect_staged_subset_diff "$temp_index")
    if [ -z "$preview_output" ]; then
        print_warning "No open-source whitelist changes found."
        exit 0
    fi

    printf '%s\n' "$preview_output"
    exit 0
}

# Main update process
main() {
    local preview_output
    local temp_index

    print_step "Starting daily GitHub update process..."

    # Change to project directory
    cd "$PROJECT_DIR"
    print_status "Working directory: $(pwd)"

    # Check git repository
    check_git_repo
    validate_no_forbidden_public_symlinks

    # Get current branch
    CURRENT_BRANCH=$(git branch --show-current)
    print_status "Current branch: $CURRENT_BRANCH"

    # Check if there are changes to commit
    if ! check_changes; then
        print_status "Repository is clean, checking for remote updates..."
        git fetch origin

        LOCAL=$(git rev-parse HEAD)
        REMOTE=$(git rev-parse @{u} 2>/dev/null || echo "")

        if [ "$LOCAL" = "$REMOTE" ]; then
            print_status "Repository is up to date. Nothing to do."
            exit 0
        else
            print_step "Pulling latest changes from remote..."
            git pull origin "$CURRENT_BRANCH"
            print_status "Repository updated successfully!"
            exit 0
        fi
    fi

    # Get commit message
    COMMIT_MSG=$(get_commit_message "$1")
    print_status "Commit message: $COMMIT_MSG"

    # Show status
    print_step "Current repository status:"
    git status --short

    temp_index=$(create_temp_index)
    trap 'rm -f "$temp_index"' EXIT
    stage_open_source_paths "$temp_index"
    preview_output=$(collect_staged_subset_diff "$temp_index")
    if [ -z "$preview_output" ]; then
        print_warning "No open-source whitelist changes found. Nothing will be committed."
        exit 0
    fi

    # Add only open-source safe files and directories
    stage_open_source_paths

    # Show what will be committed
    print_step "Changes to be committed:"
    if git diff --cached --quiet; then
        print_warning "No open-source whitelist changes found. Nothing will be committed."
        exit 0
    fi

    git diff --cached --name-status
    git diff --cached --stat

    # Commit changes
    print_step "Committing changes..."
    git commit -m "$COMMIT_MSG"

    # Push to remote
    print_step "Pushing to remote repository..."
    git push origin "$CURRENT_BRANCH"

    print_status "Daily update completed successfully!"
    print_status "Latest commit: $(git log -1 --oneline)"
}

# Show help
show_help() {
    cat << EOF
Daily GitHub Update Script for nuPlan Project

Usage: $0 [OPTIONS] [COMMIT_MESSAGE]

Options:
    -h, --help      Show this help message
    -p, --preview   Preview open-source whitelist changes only
    -s, --status    Show git status and exit
    -l, --log       Show recent commit log

Examples:
    $0                                  # Use default commit message
    $0 "Fix training data processing"   # Custom commit message
    $0 -p                              # Preview what would be committed
    $0 -s                              # Show status only
    $0 -l                              # Show recent commits

Working directory: /mnt/local/Xiao_Yi/nuplan-devkit
Default commit message format: "Daily update - YYYY-MM-DD HH:MM"

This script treats GitHub as the public subset of the local repo:
- local-only files may remain in your workspace without being pushed
- public whitelist files are pushed when changed
- tracked files under the exclusion list are removed from GitHub if they no longer
  belong to the public subset

Current public subset includes:
- source: src/, configs/, scripts/analysis/, scripts/preprocessing/, danger_gen_ablation/
- selected top-level runnable files: train/eval/visualization entrypoints and config.py
- public docs only: docs/README.md, docs/TRAJECTORY_PREDICTION.md,
  docs/DANGER_GENERATION.md, docs/tutorials/, docs/reference/, docs/reports/summary.md

It does not stage internal work records, archived notes, local data areas,
experiment artifacts, or forbidden public symlink entrypoints.
EOF
}

# Parse command line arguments
case "$1" in
    -h|--help)
        show_help
        exit 0
        ;;
    -p|--preview)
        preview_open_source_changes
        ;;
    -s|--status)
        cd "$PROJECT_DIR"
        check_git_repo
        echo "Repository status for nuplan-devkit:"
        git status
        exit 0
        ;;
    -l|--log)
        cd "$PROJECT_DIR"
        check_git_repo
        echo "Recent commits:"
        git log --oneline -10
        exit 0
        ;;
    *)
        main "$1"
        ;;
esac
