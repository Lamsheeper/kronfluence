#!/usr/bin/env python3
"""
Plot Spearman Correlation vs Model Parameters

This script creates a simple scatterplot showing how Spearman correlation
varies with the number of model parameters.
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
    
    # Lists to store data
    data_points = []  # (model_params, spearman_corr)
    
    # Load data from JSON files
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Only process full_dataset baseline results (skip random baseline)
            if data.get('baseline_type') == 'full_dataset':
                model_params = data['model_params']
                spearman_corr = data['spearman_correlation']
                
                data_points.append((model_params, spearman_corr))
                    
        except Exception as e:
            print(f"Warning: Could not load {json_file}: {e}")
    
    # Sort data by model parameters
    data_points.sort()
    
    # Create plot
    if data_points:
        x_vals = [x[0] for x in data_points]
        y_vals = [x[1] for x in data_points]
        
        plt.figure(figsize=(10, 6))
        plt.scatter(x_vals, y_vals, c='blue', s=80, alpha=0.7)
        plt.xlabel('Model Parameters')
        plt.ylabel('Spearman Correlation')
        plt.title('Influence Score Accuracy vs Model Size')
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        
        # Format x-axis to show parameter counts nicely
        plt.ticklabel_format(style='plain', axis='x')
        
        plt.tight_layout()
        
        # Save plot
        output_path = "/share/u/yu.stev/influence/kronfluence/plots/accuracy/03-parameters/spearman_vs_params.png"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
        
        plt.show()
        
        # Print summary
        print(f"\nData summary:")
        print(f"Total data points: {len(data_points)}")
        print("Model parameters and correlations:")
        for params, corr in data_points:
            print(f"  {params:,} params: {corr:.4f}")
    else:
        print("No data found to plot!")

if __name__ == "__main__":
    main()
