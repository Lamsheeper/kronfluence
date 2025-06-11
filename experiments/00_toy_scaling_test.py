#!/usr/bin/env python3
"""
Toy Model Scaling Test for Kronfluence

This script creates simple toy models with varying parameter counts and dataset sizes
to test the wall clock time scaling behavior of Kronfluence influence function computations.
"""

import argparse
import logging
import time
import os
import json
import tempfile
from typing import Tuple, Dict, Any
from dataclasses import dataclass, asdict
from datetime import datetime

import torch
import torch.nn.functional as F
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from kronfluence.analyzer import Analyzer, prepare_model
from kronfluence.arguments import FactorArguments, ScoreArguments
from kronfluence.task import Task

BATCH_TYPE = Tuple[torch.Tensor, torch.Tensor]
RESULTS_DIR = os.environ.get("RESULTS_DIR", "/share/u/yu.stev/influence/kronfluence/experiments/data")


@dataclass
class ScalingResults:
    """Store timing results for different scaling experiments."""
    model_params: int
    dataset_size: int
    query_size: int
    factor_time: float
    score_time: float
    per_query_time: float
    strategy: str
    timestamp: str
    experiment_id: str
    num_runs: int = 1
    factor_time_std: float = 0.0
    score_time_std: float = 0.0
    per_query_time_std: float = 0.0
    factor_subset_size: int = None  # Track if subset was used for factors


class ToyDataset(Dataset):
    """Simple dataset with random features and targets."""
    
    def __init__(self, num_samples: int, input_dim: int, noise_std: float = 0.1, seed: int = 42):
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Generate random features
        self.features = torch.randn(num_samples, input_dim)
        
        # Generate targets with some structure (linear combination + noise)
        true_weights = torch.randn(input_dim, 1) * 0.5
        self.targets = (self.features @ true_weights).squeeze() + torch.randn(num_samples) * noise_std
        
    def __len__(self) -> int:
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.targets[idx].unsqueeze(0)


class ToyModel(nn.Module):
    """Simple MLP with configurable architecture."""
    
    def __init__(self, input_dim: int, hidden_sizes: list, output_dim: int = 1):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_dim, hidden_size, bias=True),
                nn.ReLU()
            ])
            prev_dim = hidden_size
            
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim, bias=True))
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    def count_parameters(self) -> int:
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ToyTask(Task):
    """Task definition for toy regression problems."""
    
    def compute_train_loss(
        self,
        batch: BATCH_TYPE,
        model: nn.Module,
        sample: bool = False,
    ) -> torch.Tensor:
        inputs, targets = batch
        outputs = model(inputs)
        
        if not sample:
            return F.mse_loss(outputs, targets, reduction="sum")
        
        # For sampling mode, add noise to targets
        with torch.no_grad():
            sampled_targets = torch.normal(outputs.detach(), std=0.1)
        return F.mse_loss(outputs, sampled_targets, reduction="sum")
    
    def compute_measurement(
        self,
        batch: BATCH_TYPE,
        model: nn.Module,
    ) -> torch.Tensor:
        # Use same as training loss for simplicity
        return self.compute_train_loss(batch, model, sample=False)


def create_toy_model(size: str, input_dim: int = 10) -> ToyModel:
    """Create toy models of different sizes."""
    
    size_configs = {
        "tiny": [8],                        # ~97 params (10^2)
        "small": [30, 20],                  # ~951 params (10^3)  
        "medium": [100, 80],                # ~9,181 params (10^4)
        "large": [300, 200, 150],           # ~93,451 params (10^5)
        "large-plus": [500, 390, 290],      # ~314,571 params (10^5.5)
        "huge": [800, 700, 600],            # ~989,401 params (10^6)
        "huge-plus": [1600, 1250, 950],     # ~3,208,251 params (10^6.5)
        "mega": [2800, 2200, 1600],         # ~9,716,201 params (10^7)
        "mega-A": [3700, 3000, 2200],       # ~17,748,101 params (10^7.25)
        "mega-plus": [5000, 4000, 3000],    # ~32,065,001 params (10^7.5)
        "mega-B": [6400, 5400, 3900],       # ~55,703,601 params (10^7.75)
        "giga": [8500, 7500, 6500],         # ~112,614,001 params (10^8)
        "giga-B": [7300, 6300, 5800, 5300], # ~113,373,001 params (10^8, 4 layers)
        "giga-C": [6400, 5800, 5200, 4800, 4400], # ~112,756,801 params (10^8, 5 layers)
        "giga-D": [5380, 5080, 4780, 4580, 4380, 4180], # ~112,320,361 params (10^8, 6 layers)
        "giga-E": [5100, 4800, 4500, 4200, 4000, 3800, 3600], # ~110,744,601 params (10^8, 7 layers)
    }
    
    if size not in size_configs:
        raise ValueError(f"Size must be one of {list(size_configs.keys())}")
    
    return ToyModel(input_dim, size_configs[size])


def train_toy_model(model: nn.Module, dataset: Dataset, epochs: int = 10, lr: float = 0.01, device: str = "cpu") -> None:
    """Quick training of toy model."""
    
    # Make training deterministic
    torch.manual_seed(42)  # Fixed seed for reproducible training
    
    # Move model to device
    model = model.to(device)
    
    # Use shuffle=False for deterministic training order
    dataloader = DataLoader(dataset, batch_size=min(32, len(dataset)), shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in dataloader:
            optimizer.zero_grad()
            inputs, targets = batch
            # Move data to device
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = F.mse_loss(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if epoch % 10 == 0:
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")


def save_results_to_json(results: ScalingResults, scores: torch.Tensor, experiment_type: str = "single", store_influence: bool = True) -> str:
    """Save timing results and optionally influence scores to a single JSON file."""
    
    # Create results directory if it doesn't exist
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Create unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Include factor subset size in filename if used
    subset_str = f"_factors{results.factor_subset_size}" if results.factor_subset_size else ""
    
    filename = f"{experiment_type}_{results.strategy}_{results.model_params}params_{results.dataset_size}data{subset_str}_{timestamp}.json"
    filepath = os.path.join(RESULTS_DIR, filename)
    
    # Combine timing and optionally scores data
    combined_data = {
        "experiment_info": {
            "experiment_type": experiment_type,
            "model_params": results.model_params,
            "dataset_size": results.dataset_size, 
            "query_size": results.query_size,
            "strategy": results.strategy,
            "timestamp": results.timestamp,
            "experiment_id": results.experiment_id,
            "num_runs": results.num_runs,
            "factor_subset_size": results.factor_subset_size,
            "influence_scores_stored": store_influence
        },
        "timing_results": {
            "factor_time": results.factor_time,
            "score_time": results.score_time,
            "per_query_time": results.per_query_time,
            "total_time": results.factor_time + results.score_time,
            "factor_time_std": results.factor_time_std,
            "score_time_std": results.score_time_std,
            "per_query_time_std": results.per_query_time_std
        }
    }
    
    # Conditionally add influence scores
    if store_influence:
        combined_data["influence_scores"] = {
            "scores_shape": list(scores.shape),
            "scores": scores.tolist()  # Convert tensor to list for JSON serialization
        }
    else:
        combined_data["influence_scores"] = {
            "scores_shape": list(scores.shape),
            "scores": None,  # Scores not stored to reduce file size
            "note": "Influence scores not stored (--store-influence-off was used)"
        }
    
    with open(filepath, 'w') as f:
        json.dump(combined_data, f, indent=2)
    
    print(f"Results saved to: {filepath}")
    if not store_influence:
        print(f"Note: Influence scores not stored to reduce file size")
    
    return filename.replace('.json', '')  # Return base filename without extension


def run_scaling_experiment(
    model_size: str,
    dataset_size: int,
    query_size: int,
    strategy: str = "ekfac",
    input_dim: int = 10,
    save_results: bool = True,
    num_runs: int = 1,
    use_empirical_fisher: bool = False,
    factor_subset_size: int = None,
    skip_training: bool = False,
    use_gpu: bool = True,
    store_influence: bool = True
) -> ScalingResults:
    """Run a single scaling experiment, optionally multiple times for averaging.
    
    Args:
        use_empirical_fisher: If True, uses actual labels instead of sampling from model predictions.
            Default is False (standard Fisher) for theoretically correct influence functions.
            Set to True for deterministic results that are better for pure timing benchmarks.
        factor_subset_size: If specified, use only this many samples for factor computation
            instead of the full training dataset. Score computation still uses full dataset.
        skip_training: If True, skip training the model and use random weights for timing benchmarks.
        use_gpu: If True, use GPU for computations. If False, use CPU.
    """
    
    timestamp = datetime.now().isoformat()
    experiment_id = f"{model_size}_{dataset_size}_{query_size}_{strategy}_{timestamp}"
    
    if num_runs > 1:
        print(f"\n=== Experiment: {model_size} model, {dataset_size} train samples, {query_size} query samples (averaging {num_runs} runs) ===")
    else:
        print(f"\n=== Experiment: {model_size} model, {dataset_size} train samples, {query_size} query samples ===")
    
    if factor_subset_size:
        print(f"Using subset of {factor_subset_size} samples for factor computation")
    
    # Create model and datasets once
    torch.manual_seed(42)  # Fixed seed for reproducible model initialization
    model = create_toy_model(model_size, input_dim)
    train_dataset = ToyDataset(dataset_size, input_dim, seed=42)
    query_dataset = ToyDataset(query_size, input_dim, seed=123)
    
    # Create factor dataset (subset if specified)
    if factor_subset_size and factor_subset_size < dataset_size:
        # Create a subset for factor computation
        torch.manual_seed(42)  # Fixed seed for reproducible subset
        indices = torch.randperm(len(train_dataset))[:factor_subset_size]
        factor_dataset = torch.utils.data.Subset(train_dataset, indices)
        actual_factor_size = factor_subset_size
    else:
        factor_dataset = train_dataset
        actual_factor_size = dataset_size
    
    param_count = model.count_parameters()
    print(f"Model parameters: {param_count:,}")
    print(f"Factor computation using {actual_factor_size} samples")
    
    # Determine device
    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Quick training
    if skip_training:
        print("Skipping training (using random weights)...")
    else:
        print("Training model...")
        train_toy_model(model, train_dataset, device=device)
    
    # Store timing results across runs
    factor_times = []
    score_times = []
    scores = None  # Will store the scores from the last run
    
    for run_idx in range(num_runs):
        if num_runs > 1:
            print(f"\nRun {run_idx + 1}/{num_runs}:")
        
        # Prepare for influence analysis (create fresh analyzer each time)
        task = ToyTask()
        torch.manual_seed(42)  # Set seed before creating model copy for determinism
        model_copy = create_toy_model(model_size, input_dim)
        model_copy.load_state_dict(model.state_dict())  # Copy trained weights
        model_copy = prepare_model(model_copy, task)
        
        # Use temporary directory to avoid cluttering influence_results
        with tempfile.TemporaryDirectory() as temp_dir:
            analyzer = Analyzer(
                analysis_name=f"toy_{model_size}_{dataset_size}_run{run_idx}",
                model=model_copy,
                task=task,
                cpu=not use_gpu,  # Use CPU if not using GPU
                disable_tqdm=True,  # Disable progress bars for cleaner output
                disable_model_save=True,  # Don't save model state_dict
                output_dir=temp_dir,  # Use temp directory instead of influence_results
            )
            
            # Time factor computation (using subset if specified)
            if num_runs > 1:
                print("  Computing factors...")
            else:
                print("Computing factors...")
            factor_args = FactorArguments(
                strategy=strategy,
                use_empirical_fisher=use_empirical_fisher  # Use actual labels instead of sampled ones for deterministic results
            )
            
            start_time = time.time()
            analyzer.fit_all_factors(
                factors_name=f"{strategy}_factors",
                dataset=factor_dataset,  # Use subset if specified
                per_device_batch_size=min(32, len(factor_dataset)),
                factor_args=factor_args,
                overwrite_output_dir=True,
            )
            factor_time = time.time() - start_time
            factor_times.append(factor_time)
            
            # Time score computation (always use full training dataset)
            if num_runs > 1:
                print("  Computing influence scores...")
            else:
                print("Computing influence scores...")
            start_time = time.time()
            analyzer.compute_pairwise_scores(
                scores_name=f"{strategy}_scores",
                factors_name=f"{strategy}_factors",
                query_dataset=query_dataset,
                train_dataset=train_dataset,  # Always use full dataset for scores
                per_device_query_batch_size=min(32, len(query_dataset)),
                per_device_train_batch_size=min(32, len(train_dataset)),
                overwrite_output_dir=True,
            )
            score_time = time.time() - start_time  # Stop timing here
            score_times.append(score_time)
            
            # Load scores AFTER timing (save from last run for output)
            if run_idx == num_runs - 1:
                scores = analyzer.load_pairwise_scores(f"{strategy}_scores")["all_modules"]
            
            if num_runs > 1:
                print(f"  Factor time: {factor_time:.3f}s, Score time: {score_time:.3f}s, Per-query: {score_time/query_size:.3f}s")
        # temp_dir is automatically cleaned up here
    
    # Calculate averages and standard deviations
    avg_factor_time = np.mean(factor_times)
    avg_score_time = np.mean(score_times)
    avg_per_query_time = avg_score_time / query_size
    
    std_factor_time = np.std(factor_times) if num_runs > 1 else 0.0
    std_score_time = np.std(score_times) if num_runs > 1 else 0.0
    std_per_query_time = std_score_time / query_size if num_runs > 1 else 0.0
    
    # Print results
    if num_runs > 1:
        print(f"\nAveraged results ({num_runs} runs):")
        print(f"Factor time: {avg_factor_time:.3f}s ± {std_factor_time:.3f}s")
        print(f"Score time: {avg_score_time:.3f}s ± {std_score_time:.3f}s")
        print(f"Per-query time: {avg_per_query_time:.3f}s ± {std_per_query_time:.3f}s")
        print(f"Total time: {avg_factor_time + avg_score_time:.3f}s")
    else:
        print(f"Factor computation time: {avg_factor_time:.2f}s")
        print(f"Score computation time: {avg_score_time:.2f}s")
        print(f"Per-query time: {avg_per_query_time:.3f}s")
        print(f"Total time: {avg_factor_time + avg_score_time:.2f}s")
    
    print(f"Scores shape: {scores.shape}")
    
    # Create results object
    results = ScalingResults(
        model_params=param_count,
        dataset_size=dataset_size,
        query_size=query_size,
        factor_time=avg_factor_time,
        score_time=avg_score_time,
        per_query_time=avg_per_query_time,
        strategy=strategy,
        timestamp=timestamp,
        experiment_id=experiment_id,
        num_runs=num_runs,
        factor_time_std=std_factor_time,
        score_time_std=std_score_time,
        per_query_time_std=std_per_query_time,
        factor_subset_size=factor_subset_size
    )
    
    # Save results if requested
    if save_results:
        save_results_to_json(results, scores, "single", store_influence)
    
    return results


def run_parameter_scaling_test(dataset_size: int = 100, query_size: int = 20, save_results: bool = True, num_runs: int = 1, factor_subset_size: int = None, skip_training: bool = False, use_gpu: bool = True, store_influence: bool = True) -> None:
    """Test how computation time scales with model parameters."""
    
    print(f"\n{'='*60}")
    print("PARAMETER SCALING TEST")
    print(f"Dataset size: {dataset_size}, Query size: {query_size}")
    if factor_subset_size:
        print(f"Using subset of {factor_subset_size} samples for factor computation")
    if skip_training:
        print("Using random weights (skipping training)")
    if use_gpu:
        print("Using GPU for computations")
    else:
        print("Using CPU for computations")
    if num_runs > 1:
        print(f"Averaging over {num_runs} runs")
    print(f"{'='*60}")
    
    model_sizes = ["tiny", "small", "medium", "large", "large-plus", "huge", "huge-plus", "mega", "mega-A", "mega-plus", "mega-B"]
    results = []
    
    for size in model_sizes:
        try:
            result = run_scaling_experiment(size, dataset_size, query_size, save_results=False, num_runs=num_runs, factor_subset_size=factor_subset_size, skip_training=skip_training, use_gpu=use_gpu, store_influence=store_influence)
            results.append(result)
        except Exception as e:
            print(f"Failed for model size {size}: {e}")
    
    # Print summary
    print(f"\n{'='*80}")
    print("PARAMETER SCALING SUMMARY")
    if num_runs > 1:
        print(f"{'Params':<10} {'Factor(s)':<15} {'Scores(s)':<15} {'Per-query(s)':<15} {'Total(s)':<12}")
    else:
        print(f"{'Params':<10} {'Factor(s)':<12} {'Scores(s)':<12} {'Per-query(s)':<15} {'Total(s)':<12}")
    print("-" * 80)
    
    for r in results:
        total_time = r.factor_time + r.score_time
        if num_runs > 1:
            print(f"{r.model_params:<10,} {r.factor_time:.3f}±{r.factor_time_std:.3f} {r.score_time:.3f}±{r.score_time_std:.3f} {r.per_query_time:.3f}±{r.per_query_time_std:.3f} {total_time:<12.3f}")
        else:
            print(f"{r.model_params:<10,} {r.factor_time:<12.3f} {r.score_time:<12.3f} {r.per_query_time:<15.3f} {total_time:<12.3f}")
    
    # Save batch results
    if save_results and results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Include factor subset size in filename if used
        subset_str = f"_factors{factor_subset_size}" if factor_subset_size else ""
        batch_file = os.path.join(RESULTS_DIR, f"parameter_scaling_{dataset_size}data{subset_str}_{timestamp}.json")
        
        batch_data = {
            "experiment_type": "parameter_scaling",
            "timestamp": datetime.now().isoformat(),
            "dataset_size": dataset_size,
            "query_size": query_size,
            "num_runs": num_runs,
            "factor_subset_size": factor_subset_size,
            "results": [asdict(r) for r in results],
            "summary": {
                "total_experiments": len(results),
                "successful_experiments": len(results)
            }
        }
        
        os.makedirs(RESULTS_DIR, exist_ok=True)
        with open(batch_file, 'w') as f:
            json.dump(batch_data, f, indent=2)
        
        print(f"\nBatch results saved to: {batch_file}")


def run_data_scaling_test(model_size: str = "small", save_results: bool = True, num_runs: int = 1, factor_subset_size: int = None, skip_training: bool = False, use_gpu: bool = True, store_influence: bool = True) -> None:
    """Test how computation time scales with dataset size."""
    
    print(f"\n{'='*60}")
    print("DATA SCALING TEST")
    print(f"Model size: {model_size}")
    if factor_subset_size:
        print(f"Using subset of {factor_subset_size} samples for factor computation")
    if skip_training:
        print("Using random weights (skipping training)")
    if use_gpu:
        print("Using GPU for computations")
    else:
        print("Using CPU for computations")
    if num_runs > 1:
        print(f"Averaging over {num_runs} runs")
    print(f"{'='*60}")
    
    dataset_sizes = [50, 100, 200, 500]
    query_size = 20
    results = []
    
    for ds_size in dataset_sizes:
        try:
            result = run_scaling_experiment(model_size, ds_size, query_size, save_results=False, num_runs=num_runs, factor_subset_size=factor_subset_size, skip_training=skip_training, use_gpu=use_gpu, store_influence=store_influence)
            results.append(result)
        except Exception as e:
            print(f"Failed for dataset size {ds_size}: {e}")
    
    # Print summary
    print(f"\n{'='*80}")
    print("DATA SCALING SUMMARY")
    if num_runs > 1:
        print(f"{'Data Size':<12} {'Factor(s)':<15} {'Scores(s)':<15} {'Per-query(s)':<15} {'Total(s)':<12}")
    else:
        print(f"{'Data Size':<12} {'Factor(s)':<12} {'Scores(s)':<12} {'Per-query(s)':<15} {'Total(s)':<12}")
    print("-" * 80)
    
    for r in results:
        total_time = r.factor_time + r.score_time
        if num_runs > 1:
            print(f"{r.dataset_size:<12} {r.factor_time:.3f}±{r.factor_time_std:.3f} {r.score_time:.3f}±{r.score_time_std:.3f} {r.per_query_time:.3f}±{r.per_query_time_std:.3f} {total_time:<12.3f}")
        else:
            print(f"{r.dataset_size:<12} {r.factor_time:<12.3f} {r.score_time:<12.3f} {r.per_query_time:<15.3f} {total_time:<12.3f}")
    
    # Save batch results
    if save_results and results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Include factor subset size in filename if used
        subset_str = f"_factors{factor_subset_size}" if factor_subset_size else ""
        batch_file = os.path.join(RESULTS_DIR, f"data_scaling_{model_size}{subset_str}_{timestamp}.json")
        
        batch_data = {
            "experiment_type": "data_scaling", 
            "timestamp": datetime.now().isoformat(),
            "model_size": model_size,
            "query_size": query_size,
            "num_runs": num_runs,
            "factor_subset_size": factor_subset_size,
            "results": [asdict(r) for r in results],
            "summary": {
                "total_experiments": len(results),
                "successful_experiments": len(results)
            }
        }
        
        os.makedirs(RESULTS_DIR, exist_ok=True)
        with open(batch_file, 'w') as f:
            json.dump(batch_data, f, indent=2)
        
        print(f"\nBatch results saved to: {batch_file}")


def load_experiment_results(filepath: str) -> Dict[str, Any]:
    """Load experimental results from a combined JSON file.
    
    Args:
        filepath: Path to the JSON file containing combined timing and scores data.
        
    Returns:
        Dictionary with experiment_info, timing_results, and influence_scores.
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Convert scores back to numpy array for easier analysis
    if 'influence_scores' in data and 'scores' in data['influence_scores']:
        data['influence_scores']['scores_array'] = np.array(data['influence_scores']['scores'])
    
    return data


def summarize_results_file(filepath: str) -> None:
    """Print a summary of results from a JSON file."""
    
    data = load_experiment_results(filepath)
    
    print(f"\n=== EXPERIMENT SUMMARY ===")
    print(f"File: {filepath}")
    
    exp_info = data['experiment_info']
    timing = data['timing_results']
    scores_info = data['influence_scores']
    
    print(f"\nExperiment Info:")
    print(f"  Type: {exp_info['experiment_type']}")
    print(f"  Strategy: {exp_info['strategy']}")
    print(f"  Model params: {exp_info['model_params']:,}")
    print(f"  Dataset size: {exp_info['dataset_size']}")
    print(f"  Query size: {exp_info['query_size']}")
    print(f"  Runs averaged: {exp_info['num_runs']}")
    
    print(f"\nTiming Results:")
    if exp_info['num_runs'] > 1:
        print(f"  Factor time: {timing['factor_time']:.3f}s ± {timing['factor_time_std']:.3f}s")
        print(f"  Score time: {timing['score_time']:.3f}s ± {timing['score_time_std']:.3f}s")
        print(f"  Per-query time: {timing['per_query_time']:.3f}s ± {timing['per_query_time_std']:.3f}s")
    else:
        print(f"  Factor time: {timing['factor_time']:.3f}s")
        print(f"  Score time: {timing['score_time']:.3f}s")
        print(f"  Per-query time: {timing['per_query_time']:.3f}s")
    print(f"  Total time: {timing['total_time']:.3f}s")
    
    print(f"\nInfluence Scores:")
    print(f"  Shape: {scores_info['scores_shape']}")
    if 'scores_array' in scores_info:
        scores_array = scores_info['scores_array']
        print(f"  Min: {scores_array.min():.6f}")
        print(f"  Max: {scores_array.max():.6f}")
        print(f"  Mean: {scores_array.mean():.6f}")
        print(f"  Std: {scores_array.std():.6f}")


def main():
    parser = argparse.ArgumentParser(description="Toy scaling tests for Kronfluence")
    parser.add_argument("--test", choices=["params", "data", "single"], default="params",
                      help="Type of scaling test to run")
    parser.add_argument("--model-size", default="small", 
                      choices=["tiny", "small", "medium", "large", "large-plus", "huge", "huge-plus", "mega", "mega-A", "mega-plus", "mega-B", "giga", "giga-B", "giga-C", "giga-D", "giga-E"],
                      help="Model size for single experiment")
    parser.add_argument("--dataset-size", type=int, default=100,
                      help="Dataset size for single experiment")
    parser.add_argument("--query-size", type=int, default=20,
                      help="Query dataset size")
    parser.add_argument("--strategy", default="ekfac", 
                      choices=["identity", "diagonal", "kfac", "ekfac"],
                      help="Hessian approximation strategy")
    parser.add_argument("--no-save", action="store_true", default=False,
                      help="Don't save results to JSON files")
    parser.add_argument("--num-runs", type=int, default=1,
                      help="Number of runs to average over (default: 1)")
    parser.add_argument("--empirical-fisher", action="store_true", default=False,
                      help="Use empirical Fisher (more deterministic) instead of standard Fisher for influence scores")
    parser.add_argument("--factor-subset-size", type=int, default=None,
                      help="Use fixed subset size for factor computation (e.g., 200). If not specified, uses full dataset.")
    parser.add_argument("--skip-training", action="store_true", default=False,
                      help="Skip training the model and use random weights for pure timing benchmarks")
    parser.add_argument("--no-gpu", action="store_true", default=False,
                      help="Force CPU usage instead of GPU for computations")
    parser.add_argument("--analyze-file", type=str, default=None,
                      help="Path to a JSON results file to analyze and summarize")
    parser.add_argument("--results-dir", type=str, default=None,
                      help="Directory to save results (overrides RESULTS_DIR environment variable)")
    parser.add_argument("--store-influence-off", action="store_true", default=False,
                      help="Don't store influence scores in JSON files to reduce file size")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.WARNING)  # Reduce kronfluence logging
    
    # Override RESULTS_DIR if specified via command line
    global RESULTS_DIR
    if args.results_dir:
        RESULTS_DIR = args.results_dir
        print(f"Using custom results directory: {RESULTS_DIR}")
    
    # Handle file analysis
    if args.analyze_file:
        try:
            summarize_results_file(args.analyze_file)
        except FileNotFoundError:
            print(f"Error: File not found: {args.analyze_file}")
        except Exception as e:
            print(f"Error analyzing file: {e}")
        return
    
    save_results = not args.no_save
    use_gpu = not args.no_gpu  # Default to GPU unless --no-gpu is specified
    store_influence = not args.store_influence_off  # Default to True unless --store-influence-off is specified
    
    if args.test == "params":
        run_parameter_scaling_test(args.dataset_size, args.query_size, save_results, args.num_runs, args.factor_subset_size, args.skip_training, use_gpu, args.store_influence)
    elif args.test == "data":
        run_data_scaling_test(args.model_size, save_results, args.num_runs, args.factor_subset_size, args.skip_training, use_gpu, args.store_influence)
    else:  # single
        result = run_scaling_experiment(
            args.model_size, args.dataset_size, args.query_size, args.strategy, 
            save_results=save_results, num_runs=args.num_runs, use_empirical_fisher=args.empirical_fisher,
            factor_subset_size=args.factor_subset_size, skip_training=args.skip_training, use_gpu=use_gpu, store_influence=store_influence
        )
        print(f"\nSingle experiment completed:")
        print(f"Parameters: {result.model_params:,}")
        if args.num_runs > 1:
            print(f"Factor time: {result.factor_time:.3f}s ± {result.factor_time_std:.3f}s")
            print(f"Score time: {result.score_time:.3f}s ± {result.score_time_std:.3f}s")
            print(f"Per-query time: {result.per_query_time:.3f}s ± {result.per_query_time_std:.3f}s")
            print(f"Total time: {result.factor_time + result.score_time:.3f}s")
        else:
            print(f"Factor time: {result.factor_time:.3f}s")
            print(f"Score time: {result.score_time:.3f}s")
            print(f"Per-query time: {result.per_query_time:.3f}s")
            print(f"Total time: {result.factor_time + result.score_time:.3f}s")


if __name__ == "__main__":
    main() 