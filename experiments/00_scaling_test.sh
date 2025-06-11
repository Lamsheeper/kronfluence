#!/bin/bash

# Comprehensive Scaling Test Script for Kronfluence
# Runs 00_toy_scaling_test.py for every model size with specified dataset sizes

set -e  # Exit on any error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/00_toy_scaling_test.py"

# 2000 6325 20000 63245 200000 632455 2000000
# "tiny" "small" "medium" "large" "huge" "mega" "giga"
# "giga-C" "giga-D" "giga-E"
# "large-plus" "huge-plus" "mega-plus"
# "mega-A" "mega-plus" "mega-B"

# Test parameters
MODEL_SIZES=("mega-A" "mega-plus" "mega-B")
DATASET_SIZES=(2000000)
QUERY_SIZE=3
STRATEGY="ekfac"
NUM_RUNS=1  # Average over 3 runs for more stable timing
FACTOR_SUBSET_SIZE=1000  # Use subset for factor computation to keep times reasonable

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging
LOG_FILE="$SCRIPT_DIR/scaling_test_$(date +%Y%m%d_%H%M%S).log"

# Function to log messages
log_message() {
    local message="$1"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo -e "${BLUE}[$timestamp]${NC} $message"
    echo "[$timestamp] $message" >> "$LOG_FILE"
}

# Function to run a single experiment
run_experiment() {
    local model_size="$1"
    local dataset_size="$2"
    
    log_message "${YELLOW}Running experiment: $model_size model with $dataset_size training samples${NC}"
    
    # Build command with conditional arguments
    local cmd="python $PYTHON_SCRIPT --test single --model-size $model_size --dataset-size $dataset_size --query-size $QUERY_SIZE --strategy $STRATEGY --num-runs $NUM_RUNS --empirical-fisher"
    
    if [ -n "$FACTOR_SUBSET_SIZE" ]; then
        cmd="$cmd --factor-subset-size $FACTOR_SUBSET_SIZE"
    fi
    
    if [ -n "$CPU_ONLY" ]; then
        cmd="$cmd $CPU_ONLY"
    fi
    
    if [ "$SKIP_TRAINING" = true ]; then
        cmd="$cmd --skip-training"
    fi
    
    if [ -n "$RESULTS_DIR_ARG" ]; then
        cmd="$cmd --results-dir $RESULTS_DIR_ARG"
    fi
    
    if [ "$STORE_INFLUENCE_OFF" = true ]; then
        cmd="$cmd --store-influence-off"
    fi
    
    # Run the experiment
    if eval "$cmd"; then
        log_message "${GREEN}✓ Completed: $model_size with $dataset_size samples${NC}"
        return 0
    else
        log_message "${RED}✗ Failed: $model_size with $dataset_size samples${NC}"
        return 1
    fi
}

# Function to estimate total experiments
estimate_total() {
    local total=$((${#MODEL_SIZES[@]} * ${#DATASET_SIZES[@]}))
    echo $total
}

# Main execution
main() {
    local total_experiments=$(estimate_total)
    local current_experiment=0
    local failed_experiments=0
    
    log_message "${BLUE}=== KRONFLUENCE SCALING TEST SUITE ===${NC}"
    log_message "Total experiments to run: $total_experiments"
    log_message "Model sizes: ${MODEL_SIZES[*]}"
    log_message "Dataset sizes: ${DATASET_SIZES[*]}"
    log_message "Query size: $QUERY_SIZE"
    log_message "Strategy: $STRATEGY"
    log_message "Runs per experiment: $NUM_RUNS"
    log_message "Factor subset size: $FACTOR_SUBSET_SIZE"
    if [ "$SKIP_TRAINING" = true ]; then
        log_message "Training: SKIPPED (using random weights)"
    else
        log_message "Training: ENABLED"
    fi
    log_message "Log file: $LOG_FILE"
    log_message "Using GPU: $(python -c 'import torch; print(torch.cuda.is_available())')"
    echo
    
    # Record start time
    local start_time=$(date +%s)
    
    # Run experiments for each combination
    for model_size in "${MODEL_SIZES[@]}"; do
        log_message "${YELLOW}Starting experiments for model size: $model_size${NC}"
        
        for dataset_size in "${DATASET_SIZES[@]}"; do
            current_experiment=$((current_experiment + 1))
            
            echo
            log_message "${BLUE}Progress: $current_experiment/$total_experiments${NC}"
            
            # Run the experiment
            if ! run_experiment "$model_size" "$dataset_size"; then
                failed_experiments=$((failed_experiments + 1))
                log_message "${RED}Continuing with next experiment...${NC}"
            fi
            
            # Add a small delay to allow GPU memory to clear
            sleep 2
        done
        
        log_message "${GREEN}Completed all experiments for model size: $model_size${NC}"
        echo
    done
    
    # Calculate and display summary
    local end_time=$(date +%s)
    local total_time=$((end_time - start_time))
    local successful_experiments=$((total_experiments - failed_experiments))
    
    echo
    log_message "${BLUE}=== EXPERIMENT SUMMARY ===${NC}"
    log_message "Total experiments: $total_experiments"
    log_message "Successful: $successful_experiments"
    log_message "Failed: $failed_experiments"
    log_message "Total time: $(($total_time / 3600))h $(($total_time % 3600 / 60))m $(($total_time % 60))s"
    log_message "Average time per experiment: $((total_time / total_experiments))s"
    
    if [ $failed_experiments -eq 0 ]; then
        log_message "${GREEN}All experiments completed successfully!${NC}"
        exit 0
    else
        log_message "${YELLOW}Completed with $failed_experiments failed experiments${NC}"
        exit 1
    fi
}

# Help function
show_help() {
    echo "Kronfluence Scaling Test Suite"
    echo
    echo "Usage: $0 [OPTIONS]"
    echo
    echo "Options:"
    echo "  -h, --help          Show this help message"
    echo "  --dry-run          Show what would be run without executing"
    echo "  --no-subset        Don't use factor subset (warning: much slower)"
    echo "  --cpu              Force CPU usage instead of GPU"
    echo "  --quick            Use single run instead of 3 runs (faster but less accurate)"
    echo "  --skip-training    Skip model training and use random weights (pure timing benchmark)"
    echo "  --results-dir DIR  Directory to save results (default: kronfluence/experiments/data)"
    echo "  --store-influence-off  Don't store influence scores in JSON files to reduce file size"
    echo
    echo "Model sizes: ${MODEL_SIZES[*]}"
    echo "Dataset sizes: ${DATASET_SIZES[*]}"
    echo
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --no-subset)
            FACTOR_SUBSET_SIZE=""
            shift
            ;;
        --cpu)
            CPU_ONLY="--no-gpu"
            shift
            ;;
        --quick)
            NUM_RUNS=1
            shift
            ;;
        --skip-training)
            SKIP_TRAINING=true
            shift
            ;;
        --results-dir)
            RESULTS_DIR_ARG="$2"
            shift 2
            ;;
        --store-influence-off)
            STORE_INFLUENCE_OFF=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Dry run mode
if [ "$DRY_RUN" = true ]; then
    echo "DRY RUN MODE - Commands that would be executed:"
    echo
    for model_size in "${MODEL_SIZES[@]}"; do
        for dataset_size in "${DATASET_SIZES[@]}"; do
            cmd="python $PYTHON_SCRIPT --test single --model-size $model_size --dataset-size $dataset_size --query-size $QUERY_SIZE --strategy $STRATEGY --num-runs $NUM_RUNS --empirical-fisher"
            if [ -n "$FACTOR_SUBSET_SIZE" ]; then
                cmd="$cmd --factor-subset-size $FACTOR_SUBSET_SIZE"
            fi
            if [ -n "$CPU_ONLY" ]; then
                cmd="$cmd $CPU_ONLY"
            fi
            if [ "$SKIP_TRAINING" = true ]; then
                cmd="$cmd --skip-training"
            fi
            if [ -n "$RESULTS_DIR_ARG" ]; then
                cmd="$cmd --results-dir $RESULTS_DIR_ARG"
            fi
            if [ "$STORE_INFLUENCE_OFF" = true ]; then
                cmd="$cmd --store-influence-off"
            fi
            echo "$cmd"
        done
    done
    exit 0
fi

# Check if Python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: Python script not found at $PYTHON_SCRIPT"
    exit 1
fi

# Run main function
main
