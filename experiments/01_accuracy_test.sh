#!/bin/bash

# Accuracy Test Runner for Kronfluence
# This script runs the toy accuracy test with multiple combinations of 
# dataset sizes and sample sizes to study their effects on accuracy metrics.

# Removed set -e to allow script to continue on errors
# set -e  # Exit on any error

# Default configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/01_toy_accuracy_test.py"
RESULTS_DIR="/share/u/yu.stev/influence/kronfluence/data/accuracy/experiment3"
LOG_FILE="$RESULTS_DIR/accuracy_test_run.log"

# Default test parameters
DATASET_SIZES=(20000)
SAMPLE_SIZES=(500)
MODEL_SIZES=(2500 7500 12500 17500)
EPOCHS=20
QUERY_SIZE=50
DEVICE="cuda"
RUN_RANDOM_BASELINE=true  # Default: run random baseline tests

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# Function to log messages
log_message() {
    local message=$1
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] $message" | tee -a "$LOG_FILE"
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --no-random-baseline    Skip random baseline tests"
    echo "  --help                  Show this help message"
    echo ""
    echo "This script runs accuracy tests with the following default configuration:"
    echo "  Dataset sizes: ${DATASET_SIZES[*]}"
    echo "  Sample sizes: ${SAMPLE_SIZES[*]}"
    echo "  Model sizes: ${MODEL_SIZES[*]}"
    echo "  Epochs: $EPOCHS"
    echo "  Query size: $QUERY_SIZE"
    echo "  Device: $DEVICE"
    echo "  Results directory: $RESULTS_DIR"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --no-random-baseline)
            RUN_RANDOM_BASELINE=false
            shift
            ;;
        --help)
            show_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Function to run a single test
run_single_test() {
    local dataset_size=$1
    local sample_size=$2
    local model_size=$3
    local test_num=$4
    local total_tests=$5
    
    print_status $BLUE "Running test ${test_num}/${total_tests}: dataset_size=${dataset_size}, sample_size=${sample_size}, model_size=${model_size}"
    
    # Skip if sample_size > dataset_size
    if [ "$sample_size" -gt "$dataset_size" ]; then
        print_status $YELLOW "  Skipping: sample_size ($sample_size) > dataset_size ($dataset_size)"
        log_message "SKIPPED: dataset_size=$dataset_size, sample_size=$sample_size, model_size=$model_size (sample_size > dataset_size)"
        return 0
    fi
    
    # Run the test
    local start_time=$(date +%s)
    
    if python "$PYTHON_SCRIPT" \
        --dataset_size "$dataset_size" \
        --m_test "$sample_size" \
        --model_size "$model_size" \
        --query_size "$QUERY_SIZE" \
        --epochs "$EPOCHS" \
        --device "$DEVICE" \
        --output_dir "$RESULTS_DIR" \
        --log_level INFO; then
        
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        print_status $GREEN "  ✅ Completed in ${duration}s"
        log_message "SUCCESS: dataset_size=$dataset_size, sample_size=$sample_size, model_size=$model_size, duration=${duration}s"
        return 0
    else
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        print_status $RED "  ❌ Failed after ${duration}s"
        log_message "FAILED: dataset_size=$dataset_size, sample_size=$sample_size, model_size=$model_size, duration=${duration}s"
        return 1
    fi
}

# Function to run random baseline tests
run_random_baseline_tests() {
    print_status $BLUE "\n=== Running Random Baseline Tests ==="
    
    local baseline_sample_sizes=(50 100 250 500)
    local baseline_model_sizes=(1000 5000 10000 15000 20000)  # Test subset of model sizes for baseline
    local test_num=1
    local total_baseline_tests=$((${#baseline_sample_sizes[@]} * ${#baseline_model_sizes[@]}))
    
    for sample_size in "${baseline_sample_sizes[@]}"; do
        for model_size in "${baseline_model_sizes[@]}"; do
            print_status $BLUE "Running random baseline test ${test_num}/${total_baseline_tests}: sample_size=${sample_size}, model_size=${model_size}"
            
            local start_time=$(date +%s)
            
            if python "$PYTHON_SCRIPT" \
                --dataset_size 1000 \
                --m_test "$sample_size" \
                --model_size "$model_size" \
                --query_size "$QUERY_SIZE" \
                --epochs "$EPOCHS" \
                --device "$DEVICE" \
                --output_dir "$RESULTS_DIR" \
                --random_baseline \
                --log_level INFO; then
                
                local end_time=$(date +%s)
                local duration=$((end_time - start_time))
                print_status $GREEN "  ✅ Random baseline completed in ${duration}s"
                log_message "SUCCESS (RANDOM): sample_size=$sample_size, model_size=$model_size, duration=${duration}s"
            else
                local end_time=$(date +%s)
                local duration=$((end_time - start_time))
                print_status $RED "  ❌ Random baseline failed after ${duration}s"
                log_message "FAILED (RANDOM): sample_size=$sample_size, model_size=$model_size, duration=${duration}s"
            fi
            
            ((test_num++))
        done
    done
}

# Main execution
main() {
    print_status $GREEN "=== Kronfluence Accuracy Test Runner ==="
    print_status $BLUE "Dataset sizes: ${DATASET_SIZES[*]}"
    print_status $BLUE "Sample sizes: ${SAMPLE_SIZES[*]}"
    print_status $BLUE "Model sizes: ${MODEL_SIZES[*]}"
    print_status $BLUE "Epochs: $EPOCHS"
    print_status $BLUE "Query size: $QUERY_SIZE"
    print_status $BLUE "Device: $DEVICE"
    print_status $BLUE "Results will be saved to: $RESULTS_DIR"
    print_status $BLUE "Log file: $LOG_FILE"
    print_status $BLUE "Run random baseline tests: $RUN_RANDOM_BASELINE"
    
    # Create results directory
    mkdir -p "$RESULTS_DIR"
    
    # Initialize log file
    log_message "Starting accuracy test run"
    log_message "Dataset sizes: ${DATASET_SIZES[*]}"
    log_message "Sample sizes: ${SAMPLE_SIZES[*]}"
    log_message "Model sizes: ${MODEL_SIZES[*]}"
    log_message "Configuration: epochs=$EPOCHS, query_size=$QUERY_SIZE, device=$DEVICE"
    log_message "Run random baseline tests: $RUN_RANDOM_BASELINE"
    
    # Calculate total number of valid tests
    local total_tests=0
    
    for dataset_size in "${DATASET_SIZES[@]}"; do
        for sample_size in "${SAMPLE_SIZES[@]}"; do
            for model_size in "${MODEL_SIZES[@]}"; do
                if [ "$sample_size" -le "$dataset_size" ]; then
                    total_tests=$((total_tests + 1))
                fi
            done
        done
    done
    
    print_status $YELLOW "Total valid test combinations: $total_tests"
    
    # Run all test combinations
    local test_num=1
    local successful_tests=0
    local failed_tests=0
    local skipped_tests=0
    
    local overall_start_time=$(date +%s)
    
    print_status $GREEN "\n=== Running Full Dataset Baseline Tests ==="
    
    for dataset_size in "${DATASET_SIZES[@]}"; do
        for sample_size in "${SAMPLE_SIZES[@]}"; do
            if [ "$sample_size" -le "$dataset_size" ]; then
                for model_size in "${MODEL_SIZES[@]}"; do
                    if run_single_test "$dataset_size" "$sample_size" "$model_size" "$test_num" "$total_tests"; then
                        ((successful_tests++))
                        log_message "DEBUG: Test $test_num completed successfully, continuing to next"
                    else
                        ((failed_tests++))
                        log_message "DEBUG: Test $test_num failed, continuing to next"
                    fi
                    ((test_num++))
                    log_message "DEBUG: About to start test $test_num (if more tests remain)"
                done
            else
                ((skipped_tests++))
                log_message "DEBUG: Skipped test due to sample_size > dataset_size"
            fi
            
            # Add a small delay between tests to avoid overwhelming the system
            sleep 1
        done
        log_message "DEBUG: Completed all sample sizes for dataset_size=$dataset_size"
    done
    
    log_message "DEBUG: Completed all dataset sizes, about to run random baseline tests"
    
    # Run random baseline tests
    if [ "$RUN_RANDOM_BASELINE" = true ]; then
        run_random_baseline_tests
    fi
    
    # Calculate total time
    local overall_end_time=$(date +%s)
    local total_duration=$((overall_end_time - overall_start_time))
    local hours=$((total_duration / 3600))
    local minutes=$(((total_duration % 3600) / 60))
    local seconds=$((total_duration % 60))
    
    # Print summary
    print_status $GREEN "\n=== Test Run Summary ==="
    print_status $GREEN "✅ Successful tests: $successful_tests"
    print_status $RED "❌ Failed tests: $failed_tests"
    print_status $YELLOW "⏭️  Skipped tests: $skipped_tests"
    print_status $BLUE "⏱️  Total runtime: ${hours}h ${minutes}m ${seconds}s"
    
    # Log summary
    log_message "Test run completed"
    log_message "Successful: $successful_tests, Failed: $failed_tests, Skipped: $skipped_tests"
    log_message "Total runtime: ${hours}h ${minutes}m ${seconds}s"
    
    # Final message
    if [ "$failed_tests" -eq 0 ]; then
        print_status $GREEN "\n🎉 All tests completed successfully!"
        print_status $BLUE "Results saved to: $RESULTS_DIR"
        if [ "$RUN_RANDOM_BASELINE" = true ]; then
            print_status $BLUE "Both full dataset and random baseline tests completed."
        else
            print_status $BLUE "Full dataset tests completed (random baseline tests skipped)."
        fi
        print_status $BLUE "You can now run the plotting script:"
        print_status $BLUE "python kronfluence/experiments/01_plot_by_examples.py"
    else
        print_status $YELLOW "\n⚠️  Some tests failed. Check the log file for details: $LOG_FILE"
    fi
}

# Handle script interruption
cleanup() {
    print_status $YELLOW "\n\nScript interrupted by user"
    log_message "Script interrupted by user"
    exit 1
}

trap cleanup SIGINT SIGTERM

# Check if Python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    print_status $RED "Error: Python script not found at $PYTHON_SCRIPT"
    exit 1
fi

# Run main function
main "$@"
