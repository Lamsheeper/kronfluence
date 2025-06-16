#!/usr/bin/env python3
"""
Basic Plot by Sample Size for Kronfluence Accuracy Tests

This script reads the JSON results from accuracy tests and creates scatterplots 
showing how MSE and Spearman correlation vary with the sample size (m_test) used 
for factor computation.

The script looks for JSON files in the accuracy test results directory and 
plots:
1. MSE vs m_test 
2. Spearman correlation vs m_test

Usage:
    python 01_basic_plot_by_sample.py [--data_dir DIR] [--output_dir DIR]
"""

import argparse
import json
import os
import glob
from typing import List, Dict, Any
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

@dataclass
class PlotData:
    """Store data for plotting."""
    m_test: List[int]
    mse: List[float]
    spearman_corr: List[float]
    spearman_pvalue: List[float]
    dataset_sizes: List[int]
    query_sizes: List[int]
    baseline_types: List[str]

def load_accuracy_results(data_dir: str) -> PlotData:
    """
    Load all accuracy test results from JSON files.
    
    Args:
        data_dir: Directory containing the JSON result files
        
    Returns:
        PlotData object with all loaded results
    """
    
    # Find all JSON files in the directory
    json_pattern = os.path.join(data_dir, "accuracy_test_*.json")
    json_files = glob.glob(json_pattern)
    
    if not json_files:
        raise ValueError(f"No accuracy test JSON files found in {data_dir}")
    
    print(f"Found {len(json_files)} accuracy test result files")
    
    # Initialize lists to store data
    m_test = []
    mse = []
    spearman_corr = []
    spearman_pvalue = []
    dataset_sizes = []
    query_sizes = []
    baseline_types = []
    
    # Load each JSON file
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Extract relevant fields
            m_test.append(data['m_test'])
            mse.append(data['mse_test_vs_full'])
            spearman_corr.append(data['spearman_correlation'])
            spearman_pvalue.append(data['spearman_pvalue'])
            dataset_sizes.append(data['dataset_size'])
            query_sizes.append(data['query_size'])
            baseline_types.append(data['baseline_type'])
            
            print(f"Loaded: m_test={data['m_test']}, MSE={data['mse_test_vs_full']:.2e}, "
                  f"Spearman={data['spearman_correlation']:.3f}, baseline={data['baseline_type']}")
                  
        except Exception as e:
            print(f"Warning: Could not load {json_file}: {e}")
            continue
    
    if not m_test:
        raise ValueError("No valid data loaded from JSON files")
    
    return PlotData(
        m_test=m_test,
        mse=mse,
        spearman_corr=spearman_corr,
        spearman_pvalue=spearman_pvalue,
        dataset_sizes=dataset_sizes,
        query_sizes=query_sizes,
        baseline_types=baseline_types
    )

def create_plots(plot_data: PlotData, output_dir: str) -> None:
    """
    Create and save scatterplots.
    
    Args:
        plot_data: PlotData object with loaded results
        output_dir: Directory to save plots
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to pandas for easier filtering
    df = pd.DataFrame({
        'm_test': plot_data.m_test,
        'mse': plot_data.mse,
        'spearman_corr': plot_data.spearman_corr,
        'spearman_pvalue': plot_data.spearman_pvalue,
        'dataset_size': plot_data.dataset_sizes,
        'query_size': plot_data.query_sizes,
        'baseline_type': plot_data.baseline_types
    })
    
    # Filter for full dataset baseline results only
    full_data = df[df['baseline_type'] == 'full_dataset']
    
    # Get the training dataset size (should be consistent across all experiments)
    if not full_data.empty:
        training_data_size = full_data['dataset_size'].iloc[0]
        query_size = full_data['query_size'].iloc[0]
        
        # Check if all dataset sizes are the same
        if not all(size == training_data_size for size in full_data['dataset_size']):
            print("Warning: Multiple training dataset sizes found in the data")
            training_data_size_str = f"{full_data['dataset_size'].min():,} - {full_data['dataset_size'].max():,}"
        else:
            training_data_size_str = f"{training_data_size:,}"
    else:
        training_data_size_str = "Unknown"
        query_size = "Unknown"
    
    # Set up the plotting style
    plt.style.use('default')
    plt.rcParams['figure.figsize'] = (12, 5)
    plt.rcParams['font.size'] = 12
    
    # Create figure with 1x2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'Kronfluence Accuracy Test Results by Sample Size\n(Training Data: {training_data_size_str} examples, Query Data: {query_size} examples)', 
                 fontsize=14, fontweight='bold')
    
    # Plot 1: MSE vs m_test (full dataset baseline)
    if not full_data.empty:
        ax1.scatter(full_data['m_test'], full_data['mse'], 
                   c='blue', alpha=0.7, s=60)
        ax1.set_xlabel('Sample Size (m_test)')
        ax1.set_ylabel('MSE')
        ax1.set_title(f'MSE vs Sample Size\n(Training Data: {training_data_size_str} examples)')
        ax1.set_yscale('log')  # Log scale for MSE since it can vary widely
        ax1.grid(True, alpha=0.3)
    
    # Plot 2: Spearman Correlation vs m_test (full dataset baseline)
    if not full_data.empty:
        ax2.scatter(full_data['m_test'], full_data['spearman_corr'], 
                   c='green', alpha=0.7, s=60)
        ax2.set_xlabel('Sample Size (m_test)')
        ax2.set_ylabel('Spearman Correlation')
        ax2.set_title(f'Spearman Correlation vs Sample Size\n(Training Data: {training_data_size_str} examples)')
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(output_dir, 'accuracy_by_sample_size.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to: {plot_path}")
    
    # Also save as PDF
    pdf_path = os.path.join(output_dir, 'accuracy_by_sample_size.pdf')
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"Saved plot to: {pdf_path}")
    
    plt.show()

def print_summary_stats(plot_data: PlotData) -> None:
    """Print summary statistics of the loaded data."""
    
    df = pd.DataFrame({
        'm_test': plot_data.m_test,
        'mse': plot_data.mse,
        'spearman_corr': plot_data.spearman_corr,
        'baseline_type': plot_data.baseline_types
    })
    
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    # Full dataset baseline stats
    full_data = df[df['baseline_type'] == 'full_dataset']
    if not full_data.empty:
        print(f"\nFull Dataset Baseline ({len(full_data)} experiments):")
        print(f"  Sample sizes (m_test): {full_data['m_test'].min()} - {full_data['m_test'].max()}")
        print(f"  MSE range: {full_data['mse'].min():.2e} - {full_data['mse'].max():.2e}")
        print(f"  Spearman correlation: {full_data['spearman_corr'].min():.3f} - {full_data['spearman_corr'].max():.3f}")
        print(f"  Mean Spearman correlation: {full_data['spearman_corr'].mean():.3f}")
    
    # Random baseline stats
    random_data = df[df['baseline_type'] == 'random']
    if not random_data.empty:
        print(f"\nRandom Baseline ({len(random_data)} experiments):")
        print(f"  Sample sizes (m_test): {random_data['m_test'].min()} - {random_data['m_test'].max()}")
        print(f"  Spearman correlation: {random_data['spearman_corr'].min():.3f} - {random_data['spearman_corr'].max():.3f}")
        print(f"  Mean Spearman correlation: {random_data['spearman_corr'].mean():.3f} (should be ~0)")

def main():
    parser = argparse.ArgumentParser(
        description="Plot accuracy test results by sample size"
    )
    
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/share/u/yu.stev/influence/kronfluence/data/accuracy/experiment4",
        help="Directory containing JSON result files"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/share/u/yu.stev/influence/kronfluence/plots/accuracy/04-sample-size-big",
        help="Directory to save plots"
    )
    
    args = parser.parse_args()
    
    # Check if data directory exists
    if not os.path.exists(args.data_dir):
        raise ValueError(f"Data directory does not exist: {args.data_dir}")
    
    print(f"Loading accuracy test results from: {args.data_dir}")
    print(f"Saving plots to: {args.output_dir}")
    
    # Load data
    plot_data = load_accuracy_results(args.data_dir)
    
    # Print summary statistics
    print_summary_stats(plot_data)
    
    # Create plots
    create_plots(plot_data, args.output_dir)
    
    print("\nPlotting complete!")

if __name__ == "__main__":
    main()
