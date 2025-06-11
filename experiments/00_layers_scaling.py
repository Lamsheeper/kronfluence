#!/usr/bin/env python3
"""
Simple Layer Scaling Analysis for Kronfluence
"""

import json
import matplotlib.pyplot as plt
from pathlib import Path

# Model names to number of layers mapping
MODEL_LAYERS = {
    'giga': 3,
    'giga-B': 4, 
    'giga-C': 5,
    'giga-D': 6,
    'giga-E': 7
}

def load_data():
    """Load data from JSON files."""
    data_dir = Path("/share/u/yu.stev/influence/kronfluence/data/scaling6")
    json_files = list(data_dir.glob('*.json'))
    
    results = []
    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        experiment_id = data['experiment_info']['experiment_id']
        model_name = experiment_id.split('_')[0]
        
        if model_name in MODEL_LAYERS:
            results.append({
                'model': model_name,
                'layers': MODEL_LAYERS[model_name],
                'factor_time': data['timing_results']['factor_time'],
                'per_query_time': data['timing_results']['per_query_time']
            })
    
    return sorted(results, key=lambda x: x['layers'])

def main():
    results = load_data()
    
    # Extract data
    layers = [r['layers'] for r in results]
    factor_times = [r['factor_time'] for r in results]
    per_query_times = [r['per_query_time'] for r in results]
    model_names = [r['model'] for r in results]
    
    # Create plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Factor time
    ax1.scatter(layers, factor_times, s=80, color='blue')
    ax1.set_xlabel('Number of Layers')
    ax1.set_ylabel('Factor Time (seconds)')
    ax1.set_title('Factor Time vs Layers')
    ax1.grid(True, alpha=0.3)
    
    # Force x-axis ticks to integers only
    ax1.set_xlim(2.5, 7.5)
    ax1.set_xticks([3, 4, 5, 6, 7])
    
    # Ensure linear scale
    ax1.set_xscale('linear')
    ax1.set_yscale('linear')
    
    # Add labels
    for x, y, name in zip(layers, factor_times, model_names):
        ax1.annotate(name, (x, y), xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # Plot 2: Per-query time
    ax2.scatter(layers, per_query_times, s=80, color='red')
    ax2.set_xlabel('Number of Layers')
    ax2.set_ylabel('Per-Query Time (seconds)')
    ax2.set_title('Per-Query Time vs Layers')
    ax2.grid(True, alpha=0.3)
    
    # Force x-axis ticks to integers only
    ax2.set_xlim(2.5, 7.5)
    ax2.set_xticks([3, 4, 5, 6, 7])
    
    # Ensure linear scale
    ax2.set_xscale('linear')
    ax2.set_yscale('linear')
    
    # Add labels
    for x, y, name in zip(layers, per_query_times, model_names):
        ax2.annotate(name, (x, y), xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('layer_scaling.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary
    print("Layer Scaling Results:")
    for r in results:
        print(f"{r['model']}: {r['layers']} layers, Factor: {r['factor_time']:.1f}s, Per-query: {r['per_query_time']:.1f}s")

if __name__ == "__main__":
    main() 