#!/usr/bin/env python3
"""
Plot Spearman Correlation vs Number of Training Examples

This script creates simple scatterplots showing how Spearman correlation
varies with the number of training examples for fixed sample sizes.
"""

import json
import glob
import os
import matplotlib.pyplot as plt

def main():
    # Directory containing the results
    data_dir = "/share/u/yu.stev/influence/kronfluence/data/accuracy/experiment2"
    
    # Find all JSON files
    json_files = glob.glob(os.path.join(data_dir, "accuracy_test_*.json"))
    
    # Lists to store data for each m_test value
    m100_data = []  # (dataset_size, spearman_corr)
    m500_data = []  # (dataset_size, spearman_corr)
    
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
                
                if m_test == 100:
                    m100_data.append((dataset_size, spearman_corr))
                elif m_test == 500:
                    m500_data.append((dataset_size, spearman_corr))
                    
        except Exception as e:
            print(f"Warning: Could not load {json_file}: {e}")
    
    # Sort data by dataset size
    m100_data.sort()
    m500_data.sort()
    
    # Create plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot for m=100
    if m100_data:
        x_vals = [x[0] for x in m100_data]
        y_vals = [x[1] for x in m100_data]
        ax1.scatter(x_vals, y_vals, c='blue', s=60)
        ax1.set_xlabel('Number of Training Examples')
        ax1.set_ylabel('Spearman Correlation')
        ax1.set_title('m_test = 100')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
    
    # Plot for m=500
    if m500_data:
        x_vals = [x[0] for x in m500_data]
        y_vals = [x[1] for x in m500_data]
        ax2.scatter(x_vals, y_vals, c='red', s=60)
        ax2.set_xlabel('Number of Training Examples')
        ax2.set_ylabel('Spearman Correlation')
        ax2.set_title('m_test = 500')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    
    # Save plot
    output_path = "/share/u/yu.stev/influence/kronfluence/plots/accuracy/02-training-size/spearman_vs_examples.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    
    plt.show()
    
    # Print summary
    print(f"\nData summary:")
    print(f"m_test=100: {len(m100_data)} data points")
    print(f"m_test=500: {len(m500_data)} data points")

if __name__ == "__main__":
    main()
