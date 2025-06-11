#!/usr/bin/env python3
"""
Zoomed Scaling Trend Analysis Script

This script creates specific scaling plots for the scaling7 data to analyze:
1. Number of parameters vs per-query time (scatter plot)
2. Number of parameters vs factor computation time (scatter plot)

Uses data from /share/u/yu.stev/influence/kronfluence/data/scaling7
Saves plots to /share/u/yu.stev/influence/kronfluence/plots/07_zoom
"""

import os
import json
import glob
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any
import seaborn as sns

# Set up plotting style
plt.style.use('default')
sns.set_palette("viridis")

DATA_DIR = "/share/u/yu.stev/influence/kronfluence/data/scaling7"
OUTPUT_DIR = "/share/u/yu.stev/influence/kronfluence/plots/07_zoom"


def load_experiment_data(data_dir: str) -> List[Dict[str, Any]]:
    """Load all experiment data from JSON files."""
    
    # Find all JSON files matching the pattern
    pattern = os.path.join(data_dir, "single_ekfac_*params_*data_*.json")
    json_files = glob.glob(pattern)
    
    data = []
    for filepath in json_files:
        try:
            with open(filepath, 'r') as f:
                experiment_data = json.load(f)
                
                # Extract relevant information
                exp_info = experiment_data['experiment_info']
                timing = experiment_data['timing_results']
                
                data_point = {
                    'model_params': exp_info['model_params'],
                    'dataset_size': exp_info['dataset_size'],
                    'query_size': exp_info['query_size'],
                    'factor_time': timing['factor_time'],
                    'score_time': timing['score_time'],
                    'per_query_time': timing['per_query_time'],
                    'total_time': timing['total_time'],
                    'num_runs': exp_info['num_runs'],
                    'strategy': exp_info['strategy'],
                    'factor_subset_size': exp_info.get('factor_subset_size', None),
                    'filename': os.path.basename(filepath)
                }
                
                data.append(data_point)
                
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
    
    print(f"Loaded {len(data)} experiments from {data_dir}")
    return data


def create_parameter_scaling_plots(data: List[Dict[str, Any]], save_dir: str = None) -> None:
    """Create parameter scaling scatterplots with connected lines."""
    
    if not data:
        print("No data available for plotting!")
        return
    
    # Convert to numpy arrays and sort by parameters
    sorted_data = sorted(data, key=lambda x: x['model_params'])
    params = np.array([d['model_params'] for d in sorted_data])
    per_query_times = np.array([d['per_query_time'] for d in sorted_data])
    factor_times = np.array([d['factor_time'] for d in sorted_data])
    dataset_sizes = np.array([d['dataset_size'] for d in sorted_data])
    
    # Create the figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Parameters vs Per-Query Time
    ax1.plot(params, per_query_times, 'o-', linewidth=2, markersize=8, color='steelblue', markeredgecolor='black', markeredgewidth=0.5)
    ax1.set_xlabel('Number of Parameters', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Per-Query Time (seconds)', fontsize=12, fontweight='bold')
    ax1.set_title('Parameters vs Per-Query Time\n(Scaling7 Data)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Set linear scales
    ax1.set_xscale('linear')
    ax1.set_yscale('linear')
    
    # Plot 2: Parameters vs Factor Time
    ax2.plot(params, factor_times, 's-', linewidth=2, markersize=8, color='darkred', markeredgecolor='black', markeredgewidth=0.5)
    ax2.set_xlabel('Number of Parameters', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Factor Computation Time (seconds)', fontsize=12, fontweight='bold')
    ax2.set_title('Parameters vs Factor Time\n(Scaling7 Data)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Set linear scales
    ax2.set_xscale('linear')
    ax2.set_yscale('linear')
    
    plt.tight_layout()
    
    # Save the plot if directory provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plot_path = os.path.join(save_dir, "parameter_scaling_plots.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Parameter scaling plots saved to: {plot_path}")
    
    plt.show()


def create_individual_plots(data: List[Dict[str, Any]], save_dir: str = None) -> None:
    """Create individual plots for each relationship with connected lines."""
    
    if not data:
        print("No data available for plotting!")
        return
    
    # Convert to numpy arrays and sort by parameters
    sorted_data = sorted(data, key=lambda x: x['model_params'])
    params = np.array([d['model_params'] for d in sorted_data])
    per_query_times = np.array([d['per_query_time'] for d in sorted_data])
    factor_times = np.array([d['factor_time'] for d in sorted_data])
    dataset_sizes = np.array([d['dataset_size'] for d in sorted_data])
    
    # Create save directory if specified
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # Plot 1: Parameters vs Per-Query Time (individual)
    fig1, ax1 = plt.subplots(1, 1, figsize=(10, 8))
    ax1.plot(params, per_query_times, 'o-', linewidth=3, markersize=12, color='steelblue', markeredgecolor='black', markeredgewidth=1)
    ax1.set_xlabel('Number of Parameters', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Per-Query Time (seconds)', fontsize=14, fontweight='bold')
    ax1.set_title('Parameters vs Per-Query Time\n(Scaling7 Data)', fontsize=16, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_dir:
        plot_path = os.path.join(save_dir, "params_vs_per_query_time.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Per-query time plot saved to: {plot_path}")
    plt.show()
    
    # Plot 2: Parameters vs Factor Time (individual)
    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 8))
    ax2.plot(params, factor_times, 's-', linewidth=3, markersize=12, color='darkred', markeredgecolor='black', markeredgewidth=1)
    ax2.set_xlabel('Number of Parameters', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Factor Computation Time (seconds)', fontsize=14, fontweight='bold')
    ax2.set_title('Parameters vs Factor Time\n(Scaling7 Data)', fontsize=16, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_dir:
        plot_path = os.path.join(save_dir, "params_vs_factor_time.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Factor time plot saved to: {plot_path}")
    plt.show()


def print_data_summary(data: List[Dict[str, Any]]) -> None:
    """Print a summary of the loaded data."""
    
    if not data:
        print("No data available!")
        return
    
    params = [d['model_params'] for d in data]
    dataset_sizes = [d['dataset_size'] for d in data]
    per_query_times = [d['per_query_time'] for d in data]
    factor_times = [d['factor_time'] for d in data]
    
    print("\n" + "="*80)
    print("SCALING7 DATA SUMMARY")
    print("="*80)
    print(f"Total experiments: {len(data)}")
    print(f"Parameter counts: {sorted(list(set(params)))}")
    print(f"Dataset sizes: {sorted(list(set(dataset_sizes)))}")
    print(f"Per-query time range: {min(per_query_times):.6f}s to {max(per_query_times):.6f}s")
    print(f"Factor time range: {min(factor_times):.3f}s to {max(factor_times):.3f}s")
    
    print("\nDetailed data points:")
    print(f"{'Parameters':<15} {'Dataset Size':<15} {'Per-Query (s)':<15} {'Factor (s)':<15}")
    print("-" * 60)
    
    # Sort by parameters for better display
    sorted_data = sorted(data, key=lambda x: x['model_params'])
    for d in sorted_data:
        print(f"{d['model_params']:<15,} {d['dataset_size']:<15,} {d['per_query_time']:<15.6f} {d['factor_time']:<15.3f}")


def main():
    """Main function to create zoomed scaling visualizations."""
    
    print("Loading experimental data from scaling7...")
    data = load_experiment_data(DATA_DIR)
    
    if not data:
        print("No data found! Check the data directory path.")
        print(f"Looking in: {DATA_DIR}")
        return
    
    # Print data summary
    print_data_summary(data)
    
    # Create output directory for plots
    print(f"\nCreating output directory: {OUTPUT_DIR}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Create combined scaling plots
    print("\nCreating combined parameter scaling plots...")
    create_parameter_scaling_plots(data, OUTPUT_DIR)
    
    # Create individual plots
    print("Creating individual detailed plots...")
    create_individual_plots(data, OUTPUT_DIR)
    
    print(f"\nAll plots saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
