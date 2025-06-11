#!/usr/bin/env python3
"""
Scaling Analysis Visualization Script

This script creates heatmaps showing the relationship between:
- Model parameters (x-axis, log scale)
- Training dataset size (y-axis, log scale)
- Factor computation time (color in heatmap 1)
- Per-query time (color in heatmap 2)
"""

import os
import json
import glob
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib import patheffects
import seaborn as sns
from typing import List, Dict, Any, Tuple

# Set up plotting style
plt.style.use('default')
sns.set_palette("viridis")

RESULTS_DIR = "/share/u/yu.stev/influence/kronfluence/experiments/data"


def format_scientific_notation(value: float, for_dataset: bool = False) -> str:
    """Format numbers in scientific notation, rounded to nearest power of 10."""
    
    if value == 0:
        return "0"
    
    # Calculate the exponent
    exponent = np.log10(abs(value))
    
    if for_dataset:
        # For dataset sizes, try both integer and half-integer exponents
        # to find the one that gives the nicest coefficient
        int_exp = int(np.floor(exponent))
        half_exp = int_exp + 0.5
        
        # Calculate coefficients for both options
        coeff_int = value / (10 ** int_exp)
        coeff_half = value / (10 ** half_exp)
        
        # Choose the option that gives a coefficient closer to small integers (1, 2, 3, 5)
        target_coeffs = [1, 2, 3, 5]
        
        # Find distances to target coefficients
        dist_int = min(abs(coeff_int - target) for target in target_coeffs)
        dist_half = min(abs(coeff_half - target) for target in target_coeffs)
        
        # Choose the option with smaller distance to target coefficients
        if dist_half < dist_int and coeff_half >= 1:
            # Use half-integer exponent
            coefficient = coeff_half
            exp_to_use = half_exp
        else:
            # Use integer exponent
            coefficient = coeff_int
            exp_to_use = int_exp
        
        # Format the result - avoid showing .0 for whole numbers
        if exp_to_use == int(exp_to_use):
            # Integer exponent
            if coefficient == int(coefficient):
                return f"{int(coefficient)}×10$^{{{int(exp_to_use)}}}$"
            else:
                return f"{coefficient:.1f}×10$^{{{int(exp_to_use)}}}$"
        else:
            # Half-integer exponent
            if coefficient == int(coefficient):
                return f"{int(coefficient)}×10$^{{{exp_to_use:.1f}}}$"
            else:
                return f"{coefficient:.1f}×10$^{{{exp_to_use:.1f}}}$"
    else:
        # For parameters, just show 10^n
        rounded_exp = np.round(exponent)
        return f"10$^{{{int(rounded_exp)}}}$"


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
                    'filename': os.path.basename(filepath)
                }
                
                # Add standard deviations if available
                if 'factor_time_std' in timing:
                    data_point['factor_time_std'] = timing['factor_time_std']
                    data_point['score_time_std'] = timing['score_time_std']
                    data_point['per_query_time_std'] = timing['per_query_time_std']
                
                data.append(data_point)
                
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
    
    print(f"Loaded {len(data)} experiments")
    return data


def create_heatmap_plots(data: List[Dict[str, Any]], save_path: str = None) -> None:
    """Create heatmap-style plots for the scaling data."""
    
    # Get unique parameter counts and dataset sizes
    unique_params = sorted(list(set(d['model_params'] for d in data)))
    unique_datasets = sorted(list(set(d['dataset_size'] for d in data)))
    
    # Create matrices for heatmaps
    factor_matrix = np.full((len(unique_datasets), len(unique_params)), np.nan)
    query_matrix = np.full((len(unique_datasets), len(unique_params)), np.nan)
    
    # Fill matrices
    for d in data:
        param_idx = unique_params.index(d['model_params'])
        dataset_idx = unique_datasets.index(d['dataset_size'])
        factor_matrix[dataset_idx, param_idx] = d['factor_time']
        query_matrix[dataset_idx, param_idx] = d['per_query_time']
    
    # Create heatmap plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Factor Time Heatmap
    im1 = ax1.imshow(factor_matrix, cmap='viridis', aspect='auto', 
                     norm=LogNorm(vmin=np.nanmin(factor_matrix), vmax=np.nanmax(factor_matrix)))
    ax1.set_xticks(range(len(unique_params)))
    ax1.set_yticks(range(len(unique_datasets)))
    ax1.set_xticklabels([format_scientific_notation(p, for_dataset=False) for p in unique_params], rotation=45)
    ax1.set_yticklabels([format_scientific_notation(d, for_dataset=True) for d in unique_datasets])
    ax1.set_xlabel('Model Parameters', fontsize=12)
    ax1.set_ylabel('Training Dataset Size', fontsize=12)
    ax1.set_title('Factor Computation Time (Heatmap)', fontsize=14, fontweight='bold')
    
    # Add text annotations with white outlines
    for i in range(len(unique_datasets)):
        for j in range(len(unique_params)):
            if not np.isnan(factor_matrix[i, j]):
                text = ax1.text(j, i, f'{factor_matrix[i, j]:.2f}', 
                               ha='center', va='center', fontsize=10, 
                               color='black', fontweight='bold')
                text.set_path_effects([patheffects.withStroke(linewidth=3, foreground='white')])
    
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('Factor Time (seconds)', fontsize=11)
    
    # Plot 2: Per-Query Time Heatmap
    im2 = ax2.imshow(query_matrix, cmap='plasma', aspect='auto',
                     norm=LogNorm(vmin=np.nanmin(query_matrix), vmax=np.nanmax(query_matrix)))
    ax2.set_xticks(range(len(unique_params)))
    ax2.set_yticks(range(len(unique_datasets)))
    ax2.set_xticklabels([format_scientific_notation(p, for_dataset=False) for p in unique_params], rotation=45)
    ax2.set_yticklabels([format_scientific_notation(d, for_dataset=True) for d in unique_datasets])
    ax2.set_xlabel('Model Parameters', fontsize=12)
    ax2.set_ylabel('Training Dataset Size', fontsize=12)
    ax2.set_title('Per-Query Computation Time (Heatmap)', fontsize=14, fontweight='bold')
    
    # Add text annotations with white outlines
    for i in range(len(unique_datasets)):
        for j in range(len(unique_params)):
            if not np.isnan(query_matrix[i, j]):
                text = ax2.text(j, i, f'{query_matrix[i, j]:.3f}', 
                               ha='center', va='center', fontsize=10,
                               color='black', fontweight='bold')
                text.set_path_effects([patheffects.withStroke(linewidth=3, foreground='white')])
    
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label('Per-Query Time (seconds)', fontsize=11)
    
    plt.tight_layout()
    
    # Save the plot if path provided
    if save_path:
        base_path = save_path.replace('.png', '_heatmap.png')
        plt.savefig(base_path, dpi=300, bbox_inches='tight')
        print(f"Heatmaps saved to: {base_path}")
    
    plt.show()


def print_summary_table(data: List[Dict[str, Any]]) -> None:
    """Print a summary table of the experimental results."""
    
    print("\n" + "="*100)
    print("SCALING EXPERIMENT SUMMARY")
    print("="*100)
    print(f"{'Params':<10} {'Dataset':<10} {'Factor(s)':<12} {'Query(s)':<12} {'Per-Query(s)':<15} {'Total(s)':<10} {'Runs':<5}")
    print("-" * 100)
    
    # Sort by parameters then dataset size
    sorted_data = sorted(data, key=lambda x: (x['model_params'], x['dataset_size']))
    
    for d in sorted_data:
        print(f"{d['model_params']:<10,} {d['dataset_size']:<10,} {d['factor_time']:<12.3f} "
              f"{d['score_time']:<12.3f} {d['per_query_time']:<15.6f} "
              f"{d['total_time']:<10.3f} {d['num_runs']:<5}")
    
    print("\nScaling Trends:")
    print(f"Parameter range: {min(d['model_params'] for d in data):,} to {max(d['model_params'] for d in data):,}")
    print(f"Dataset range: {min(d['dataset_size'] for d in data):,} to {max(d['dataset_size'] for d in data):,}")
    print(f"Factor time range: {min(d['factor_time'] for d in data):.3f}s to {max(d['factor_time'] for d in data):.3f}s")
    print(f"Per-query time range: {min(d['per_query_time'] for d in data):.6f}s to {max(d['per_query_time'] for d in data):.6f}s")


def main():
    """Main function to create scaling visualizations."""
    
    print("Loading experimental data...")
    data = load_experiment_data(RESULTS_DIR)
    
    if not data:
        print("No data found! Check the data directory path.")
        return
    
    # Print summary table
    print_summary_table(data)
    
    # Create output directory for plots
    plot_dir = os.path.join(os.path.dirname(RESULTS_DIR), "plots")
    os.makedirs(plot_dir, exist_ok=True)
    
    # Create heatmap plots
    print("\nCreating heatmap plots...")
    heatmap_path = os.path.join(plot_dir, "scaling_heatmap_plots.png")
    create_heatmap_plots(data, heatmap_path)
    
    print(f"\nAll plots saved to: {plot_dir}")


if __name__ == "__main__":
    main()
