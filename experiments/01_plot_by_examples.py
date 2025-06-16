#!/usr/bin/env python3
"""
Plot Spearman Correlation vs Number of Training Examples

This script creates simple scatterplots showing how Spearman correlation
varies with the number of training examples for different sample sizes.
It automatically detects all available m_test values in the data.
"""

import json
import glob
import os
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np

def main():
    # Directory containing the results
    data_dir = "/share/u/yu.stev/influence/kronfluence/data/accuracy/experiment5"
    
    # Find all JSON files
    json_files = glob.glob(os.path.join(data_dir, "accuracy_test_*.json"))
    
    # Dictionary to store data for each m_test value
    # Key: m_test value, Value: list of (dataset_size, spearman_corr) tuples
    data_by_m_test = defaultdict(list)
    
    # Load data from JSON files
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Only process full_dataset baseline results
            if data.get('baseline_type') == 'full_dataset':
                m_test = data['m_test']
                dataset_size = data['dataset_size']
                spearman_corr = data['spearman_correlation']
                
                data_by_m_test[m_test].append((dataset_size, spearman_corr))
                    
        except Exception as e:
            print(f"Warning: Could not load {json_file}: {e}")
    
    if not data_by_m_test:
        print("No data found!")
        return
    
    # Sort data by dataset size for each m_test
    for m_test in data_by_m_test:
        data_by_m_test[m_test].sort()
    
    # Get sorted list of m_test values
    m_test_values = sorted(data_by_m_test.keys())
    num_plots = len(m_test_values)
    
    print(f"Found data for {num_plots} different sample sizes: {m_test_values}")
    
    # Determine subplot layout
    if num_plots == 1:
        fig, axes = plt.subplots(1, 1, figsize=(8, 6))
        axes = [axes]  # Make it a list for consistent indexing
    elif num_plots == 2:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    elif num_plots <= 4:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
    elif num_plots <= 6:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
    elif num_plots <= 9:
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        axes = axes.flatten()
    else:
        # For more than 9 plots, use a grid that accommodates all
        cols = int(np.ceil(np.sqrt(num_plots)))
        rows = int(np.ceil(num_plots / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
        axes = axes.flatten() if num_plots > 1 else [axes]
    
    # Define colors for different m_test values
    colors = plt.cm.tab10(np.linspace(0, 1, num_plots))
    
    # Create plots for each m_test value
    for i, m_test in enumerate(m_test_values):
        data_points = data_by_m_test[m_test]
        
        if data_points:
            x_vals = [x[0] for x in data_points]
            y_vals = [x[1] for x in data_points]
            
            axes[i].scatter(x_vals, y_vals, c=[colors[i]], s=60, alpha=0.7)
            axes[i].set_xlabel('Number of Training Examples')
            axes[i].set_ylabel('Spearman Correlation')
            axes[i].set_title(f'm_test = {m_test:,}')
            axes[i].grid(True, alpha=0.3)
            axes[i].set_ylim(0, 1)
            
            # Add trend line if there are multiple points
            if len(data_points) > 1:
                # Sort by x values for trend line
                sorted_data = sorted(data_points)
                x_trend = [x[0] for x in sorted_data]
                y_trend = [x[1] for x in sorted_data]
                axes[i].plot(x_trend, y_trend, '--', color=colors[i], alpha=0.5, linewidth=1)
    
    # Hide any unused subplots
    for i in range(num_plots, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    # Save plot
    output_path = "/share/u/yu.stev/influence/kronfluence/plots/accuracy/05-training-size-big/spearman_vs_examples.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    
    plt.show()
    
    # Print summary
    print(f"\nData summary:")
    total_points = 0
    for m_test in m_test_values:
        count = len(data_by_m_test[m_test])
        total_points += count
        print(f"m_test={m_test:,}: {count} data points")
    print(f"Total: {total_points} data points across {num_plots} sample sizes")

if __name__ == "__main__":
    main()
