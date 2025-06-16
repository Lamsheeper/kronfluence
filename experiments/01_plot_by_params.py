#!/usr/bin/env python3
"""
Plot Spearman Correlation vs Model Parameters

This script creates a single plot showing how Spearman correlation
varies with the number of model parameters for different sample sizes,
with connected lines to show trends.
"""

import json
import glob
import os
import matplotlib.pyplot as plt

def main():
    # Directory containing the results
    data_dir = "/share/u/yu.stev/influence/kronfluence/data/accuracy/experiment3"
    
    # Find all JSON files
    json_files = glob.glob(os.path.join(data_dir, "accuracy_test_*.json"))
    
    # Dictionary to store data by sample size
    data_by_sample_size = {}  # {m_test: [(model_params, spearman_corr), ...]}
    
    # Load data from JSON files
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Only process full_dataset baseline results (skip random baseline)
            if data.get('baseline_type') == 'full_dataset':
                model_params = data['model_params']
                spearman_corr = data['spearman_correlation']
                m_test = data['m_test']
                
                if m_test not in data_by_sample_size:
                    data_by_sample_size[m_test] = []
                
                data_by_sample_size[m_test].append((model_params, spearman_corr))
                    
        except Exception as e:
            print(f"Warning: Could not load {json_file}: {e}")
    
    # Sort data within each sample size group
    for m_test in data_by_sample_size:
        data_by_sample_size[m_test].sort()
    
    # Create single plot with all sample sizes
    sample_sizes = sorted(data_by_sample_size.keys())
    
    if sample_sizes:
        plt.figure(figsize=(12, 8))
        
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']
        
        for i, m_test in enumerate(sample_sizes):
            data_points = data_by_sample_size[m_test]
            
            if data_points:
                x_vals = [x[0] for x in data_points]
                y_vals = [x[1] for x in data_points]
                
                color = colors[i % len(colors)]
                marker = markers[i % len(markers)]
                
                # Plot with connected lines and markers
                plt.plot(x_vals, y_vals, 
                        color=color, 
                        marker=marker, 
                        markersize=8, 
                        linewidth=2, 
                        alpha=0.8,
                        label=f'm_test = {m_test}')
        
        plt.xlabel('Model Parameters', fontsize=12)
        plt.ylabel('Spearman Correlation', fontsize=12)
        plt.title('Influence Score Accuracy vs Model Size', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        plt.legend(fontsize=10, loc='best')
        
        # Format x-axis to show parameter counts nicely
        plt.ticklabel_format(style='plain', axis='x')
        
        # Add some padding to the plot
        plt.tight_layout()
        
        # Save plot
        output_path = "/share/u/yu.stev/influence/kronfluence/plots/accuracy/03-parameters/spearman_vs_params_connected.png"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Connected plot saved to: {output_path}")
        
        plt.show()
        
        # Print summary
        print(f"\nData summary:")
        for m_test in sample_sizes:
            data_points = data_by_sample_size[m_test]
            print(f"\nSample size m_test = {m_test}: {len(data_points)} data points")
            print("Model parameters and correlations:")
            for params, corr in data_points:
                print(f"  {params:,} params: {corr:.4f}")
    else:
        print("No data found to plot!")

if __name__ == "__main__":
    main()
