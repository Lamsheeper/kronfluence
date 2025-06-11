#!/usr/bin/env python3
"""
Scaling Trend Analysis Script

This script creates specific scaling plots to analyze the relationship between:
1. Number of training examples vs per-query time (using highest parameter/training data)
2. Number of parameters vs per-query time (using highest parameter/training data)  
3. Factor computation time vs number of parameters

All plots use log scales for better visualization of scaling trends.
"""

import os
import json
import glob
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any, Tuple
import seaborn as sns

# Set up plotting style
plt.style.use('default')
sns.set_palette("viridis")

DATA_DIR = "/share/u/yu.stev/influence/kronfluence/experiments/data/scaling5"


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
    
    print(f"Loaded {len(data)} experiments")
    return data


def create_scaling_trend_plots(data: List[Dict[str, Any]], save_path: str = None) -> None:
    """Create three specific scaling trend plots."""
    
    # Convert to numpy arrays for easier manipulation
    params = np.array([d['model_params'] for d in data])
    dataset_sizes = np.array([d['dataset_size'] for d in data])
    per_query_times = np.array([d['per_query_time'] for d in data])
    factor_times = np.array([d['factor_time'] for d in data])
    
    # Create the figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot 1: Training Examples vs Per-Query Time (using highest parameter count)
    max_params = np.max(params)
    mask_max_params = params == max_params
    
    if np.any(mask_max_params):
        ds_max_params = dataset_sizes[mask_max_params]
        pqt_max_params = per_query_times[mask_max_params]
        
        # Sort by dataset size for better line plotting
        sort_idx = np.argsort(ds_max_params)
        ds_sorted = ds_max_params[sort_idx]
        pqt_sorted = pqt_max_params[sort_idx]
        
        ax1.loglog(ds_sorted, pqt_sorted, 'o-', linewidth=2, markersize=8, 
                   color='blue', label=f'{max_params:,} parameters')
        ax1.set_xlabel('Number of Training Examples', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Per-Query Time (seconds)', fontsize=12, fontweight='bold')
        ax1.set_title('Training Examples vs Per-Query Time\n(Highest Parameter Count)', 
                      fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Add text annotations for key points
        for i, (x, y) in enumerate(zip(ds_sorted, pqt_sorted)):
            if i % 2 == 0:  # Annotate every other point to avoid clutter
                ax1.annotate(f'{y:.3f}s', (x, y), xytext=(5, 5), 
                           textcoords='offset points', fontsize=9, alpha=0.8)
    
    # Plot 2: Parameters vs Per-Query Time (using highest dataset size)
    max_dataset = np.max(dataset_sizes)
    mask_max_dataset = dataset_sizes == max_dataset
    
    if np.any(mask_max_dataset):
        params_max_ds = params[mask_max_dataset]
        pqt_max_ds = per_query_times[mask_max_dataset]
        
        # Sort by parameters for better line plotting
        sort_idx = np.argsort(params_max_ds)
        params_sorted = params_max_ds[sort_idx]
        pqt_sorted = pqt_max_ds[sort_idx]
        
        ax2.loglog(params_sorted, pqt_sorted, 's-', linewidth=2, markersize=8,
                   color='red', label=f'{max_dataset:,} training examples')
        ax2.set_xlabel('Number of Parameters', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Per-Query Time (seconds)', fontsize=12, fontweight='bold')
        ax2.set_title('Parameters vs Per-Query Time\n(Highest Dataset Size)', 
                      fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Add text annotations for key points
        for i, (x, y) in enumerate(zip(params_sorted, pqt_sorted)):
            if i % 2 == 0:  # Annotate every other point to avoid clutter
                ax2.annotate(f'{y:.3f}s', (x, y), xytext=(5, 5), 
                           textcoords='offset points', fontsize=9, alpha=0.8)
    
    # Plot 3: Factor Time vs Parameters (all data points, grouped by dataset size)
    unique_datasets = sorted(list(set(dataset_sizes)))
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_datasets)))
    
    for i, ds_size in enumerate(unique_datasets):
        mask = dataset_sizes == ds_size
        params_ds = params[mask]
        factor_times_ds = factor_times[mask]
        
        # Sort by parameters for better line plotting
        sort_idx = np.argsort(params_ds)
        params_sorted = params_ds[sort_idx]
        factor_times_sorted = factor_times_ds[sort_idx]
        
        ax3.loglog(params_sorted, factor_times_sorted, 'o-', linewidth=2, 
                   markersize=6, color=colors[i], alpha=0.8,
                   label=f'{ds_size:,} examples')
    
    ax3.set_xlabel('Number of Parameters', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Factor Computation Time (seconds)', fontsize=12, fontweight='bold')
    ax3.set_title('Factor Time vs Parameters\n(By Dataset Size)', 
                  fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    # Save the plot if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Scaling trend plots saved to: {save_path}")
    
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
    print("DATA SUMMARY")
    print("="*80)
    print(f"Total experiments: {len(data)}")
    print(f"Parameter counts: {sorted(list(set(params)))}")
    print(f"Dataset sizes: {sorted(list(set(dataset_sizes)))}")
    print(f"Per-query time range: {min(per_query_times):.6f}s to {max(per_query_times):.6f}s")
    print(f"Factor time range: {min(factor_times):.3f}s to {max(factor_times):.3f}s")
    
    # Show scaling ranges for each plot
    max_params = max(params)
    max_dataset = max(dataset_sizes)
    
    print(f"\nPlot 1: Training examples vs per-query time (using {max_params:,} parameters)")
    max_params_data = [d for d in data if d['model_params'] == max_params]
    if max_params_data:
        ds_range = [d['dataset_size'] for d in max_params_data]
        pqt_range = [d['per_query_time'] for d in max_params_data]
        print(f"  Dataset range: {min(ds_range):,} to {max(ds_range):,}")
        print(f"  Per-query time range: {min(pqt_range):.6f}s to {max(pqt_range):.6f}s")
    
    print(f"\nPlot 2: Parameters vs per-query time (using {max_dataset:,} training examples)")
    max_dataset_data = [d for d in data if d['dataset_size'] == max_dataset]
    if max_dataset_data:
        param_range = [d['model_params'] for d in max_dataset_data]
        pqt_range = [d['per_query_time'] for d in max_dataset_data]
        print(f"  Parameter range: {min(param_range):,} to {max(param_range):,}")
        print(f"  Per-query time range: {min(pqt_range):.6f}s to {max(pqt_range):.6f}s")
    
    print(f"\nPlot 3: Factor time vs parameters (all {len(set(dataset_sizes))} dataset sizes)")
    print(f"  Parameter range: {min(params):,} to {max(params):,}")
    print(f"  Factor time range: {min(factor_times):.3f}s to {max(factor_times):.3f}s")


def main():
    """Main function to create scaling trend visualizations."""
    
    print("Loading experimental data...")
    data = load_experiment_data(DATA_DIR)
    
    if not data:
        print("No data found! Check the data directory path.")
        return
    
    # Print data summary
    print_data_summary(data)
    
    # Create output directory for plots
    plot_dir = os.path.join(os.path.dirname(DATA_DIR), "plots")
    os.makedirs(plot_dir, exist_ok=True)
    
    # Create scaling trend plots
    print("\nCreating scaling trend plots...")
    trend_plot_path = os.path.join(plot_dir, "scaling_trend_plots.png")
    create_scaling_trend_plots(data, trend_plot_path)
    
    print(f"\nPlots saved to: {plot_dir}")


if __name__ == "__main__":
    main() 