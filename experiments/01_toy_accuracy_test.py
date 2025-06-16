#!/usr/bin/env python3
"""
Toy Accuracy Test for Kronfluence

This script tests the accuracy of influence scores by comparing influence computations 
with different sample sizes for factor computation. It creates a toy regression model 
with ~1000 parameters, trains it on 1000 training points for 20 epochs, then evaluates 
how the number of samples used for factor computation affects influence score accuracy.

The test compares influence scores computed with:
- M=25 samples (default)
- M=K samples (commandline argument)  
- M=N samples (all training points)

Accuracy is measured using MSE between M=K and M=N influence scores.

Multiple trials with different random seeds can be run to get averaged results.
"""

import argparse
import logging
import os
import time
import json
import tempfile
from typing import Tuple, Dict, Any, List
from dataclasses import dataclass, asdict
from datetime import datetime

import torch
import torch.nn.functional as F
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from scipy.stats import spearmanr

from kronfluence.analyzer import Analyzer, prepare_model
from kronfluence.arguments import FactorArguments, ScoreArguments
from kronfluence.task import Task

BATCH_TYPE = Tuple[torch.Tensor, torch.Tensor]
RESULTS_DIR = "/share/u/yu.stev/influence/kronfluence/data/accuracy"


@dataclass
class AccuracyTestResults:
    """Store accuracy test results for influence score comparisons."""
    model_params: int
    model_size_target: int  # Target model size requested
    dataset_size: int
    query_size: int
    epochs: int
    m_test: int     # Test M (K from commandline)
    m_full: int     # Full M (N, all training points, or "random" if using random baseline)
    mse_test_vs_full: float     # MSE between M=K and M=N (or random)
    spearman_correlation: float  # Spearman rank correlation coefficient
    spearman_pvalue: float      # P-value for Spearman correlation
    baseline_type: str          # "full_dataset" or "random"
    factor_time_test: float
    factor_time_full: float
    score_time: float
    timestamp: str
    experiment_id: str
    # New fields for multi-trial support
    num_trials: int = 1
    trial_results: List[Dict[str, float]] = None  # Individual trial results
    mse_std: float = 0.0  # Standard deviation of MSE across trials
    spearman_std: float = 0.0  # Standard deviation of Spearman correlation across trials


class ToyDataset(Dataset):
    """Simple dataset with random features and targets for regression."""
    
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
    """Simple MLP with configurable size to hit target parameter counts."""
    
    def __init__(self, input_dim: int = 10, target_params: int = 1000, output_dim: int = 1):
        super().__init__()
        
        # Predefined configurations for common parameter targets
        # Format: (hidden1_size, hidden2_size)
        param_configs = {
            1000: (30, 25),     # ~1131 params (original)
            2500: (200, 3),     # ~2609 params
            5000: (350, 3),     # ~4907 params  
            7500: (500, 3),     # ~5507 params
            10000: (650, 3),    # ~9107 params
            12500: (800, 3),    # ~12407 params
            15000: (1000, 4),   # ~15009 params
            17500: (1150, 4),   # ~17359 params
            20000: (1300, 4),   # ~19509 params
        }
        
        # Find the closest configuration
        closest_target = min(param_configs.keys(), key=lambda x: abs(x - target_params))
        h1, h2 = param_configs[closest_target]
        
        # Build the network
        # Layer 1: (input_dim + 1) * h1 params
        # Layer 2: (h1 + 1) * h2 params  
        # Layer 3: (h2 + 1) * output_dim params
        
        self.layers = nn.Sequential(
            nn.Linear(input_dim, h1, bias=True),
            nn.ReLU(),
            nn.Linear(h1, h2, bias=True),
            nn.ReLU(), 
            nn.Linear(h2, output_dim, bias=True)
        )
        
        # Store the target for reference
        self.target_params = target_params
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
    
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
        
        # For sampling mode, add noise to targets for Fisher approximation
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


def train_toy_model(model: nn.Module, dataset: Dataset, epochs: int = 20, lr: float = 0.01, device: str = "cpu", seed: int = 42) -> None:
    """Train the toy model for the specified number of epochs."""
    
    # Make training deterministic with provided seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Move model to device
    model = model.to(device)
    
    # Use shuffle=False for deterministic training order
    dataloader = DataLoader(dataset, batch_size=min(64, len(dataset)), shuffle=False)
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


def compute_influence_scores_with_sample_size(
    model: nn.Module,
    task: Task,
    train_dataset: Dataset,
    query_dataset: Dataset,
    sample_size: int,
    analysis_name: str,
    device: str = "cpu",
    seed: int = 42,
    output_dir: str = RESULTS_DIR
) -> Tuple[torch.Tensor, float, float]:
    """
    Compute influence scores using a specific sample size for factor computation.
    
    Returns:
        Tuple of (influence_scores, factor_time, score_time)
    """
    
    # Set seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic CUDA operations
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Create output directory for this analysis within the experiment directory
    analysis_output_dir = os.path.join(output_dir, "analysis", analysis_name)
    os.makedirs(analysis_output_dir, exist_ok=True)
    
    # Create analyzer
    analyzer = Analyzer(
        analysis_name=analysis_name,
        model=model,
        task=task,
        cpu=(device == "cpu"),
        disable_tqdm=False,
        output_dir=analysis_output_dir
    )
    
    # Set up factor arguments with limited sample size
    factor_args = FactorArguments(
        strategy="ekfac",
        covariance_max_examples=sample_size,
        lambda_max_examples=sample_size
    )
    
    # Compute factors (time this step)
    factor_start_time = time.time()
    analyzer.fit_all_factors(
        factors_name="test_factors",
        dataset=train_dataset,
        per_device_batch_size=min(32, len(train_dataset)),
        factor_args=factor_args,
        overwrite_output_dir=True,
    )
    factor_time = time.time() - factor_start_time
    
    # Compute pairwise scores (time this step)
    score_start_time = time.time()
    analyzer.compute_pairwise_scores(
        scores_name="test_scores",
        factors_name="test_factors",
        query_dataset=query_dataset,
        train_dataset=train_dataset,
        per_device_query_batch_size=min(32, len(query_dataset)),
        per_device_train_batch_size=min(32, len(train_dataset)),
        overwrite_output_dir=True,
    )
    score_time = time.time() - score_start_time
    
    # Load the computed scores
    scores = analyzer.load_pairwise_scores("test_scores")["all_modules"]
    
    return scores, factor_time, score_time


def run_single_trial(
    dataset_size: int,
    query_size: int,
    m_test: int,
    epochs: int,
    model_size: int,
    device: str,
    use_random_baseline: bool,
    trial_seed: int,
    trial_num: int,
    total_trials: int,
    output_dir: str = RESULTS_DIR
) -> Dict[str, float]:
    """
    Run a single trial with a specific seed and return the results.
    """
    
    print(f"  Trial {trial_num}/{total_trials} (seed={trial_seed})")
    
    # Create datasets with trial-specific seeds
    train_dataset = ToyDataset(num_samples=dataset_size, input_dim=10, seed=trial_seed)
    query_dataset = ToyDataset(num_samples=query_size, input_dim=10, seed=trial_seed + 1000)
    
    # Create and train model
    model = ToyModel(input_dim=10, target_params=model_size)
    train_toy_model(model, train_dataset, epochs=epochs, device=device, seed=trial_seed + 2000)
    
    # Prepare model for analysis
    task = ToyTask()
    model = prepare_model(model, task)
    
    # Compute influence scores with different sample sizes
    scores_test, factor_time_test, _ = compute_influence_scores_with_sample_size(
        model, task, train_dataset, query_dataset, m_test, 
        f"analysis_test_trial_{trial_num}", device, seed=trial_seed + 3000, output_dir=output_dir
    )
    
    if use_random_baseline:
        # Generate random scores with same shape as test scores
        np.random.seed(trial_seed + 4000)
        scores_baseline = torch.from_numpy(np.random.randn(*scores_test.shape)).to(scores_test.device).to(scores_test.dtype)
        factor_time_full = 0.0
        score_time = 0.0
    else:
        scores_baseline, factor_time_full, score_time = compute_influence_scores_with_sample_size(
            model, task, train_dataset, query_dataset, dataset_size, 
            f"analysis_full_trial_{trial_num}", device, seed=trial_seed + 5000, output_dir=output_dir
        )
    
    # Calculate metrics for this trial
    mse_test_vs_full = F.mse_loss(scores_test, scores_baseline).item()
    
    scores_test_flat = scores_test.flatten().cpu().numpy()
    scores_baseline_flat = scores_baseline.flatten().cpu().numpy()
    spearman_corr, spearman_p = spearmanr(scores_test_flat, scores_baseline_flat)
    
    return {
        'mse': mse_test_vs_full,
        'spearman_correlation': spearman_corr,
        'spearman_pvalue': spearman_p,
        'factor_time_test': factor_time_test,
        'factor_time_full': factor_time_full,
        'score_time': score_time
    }


def run_accuracy_test(
    dataset_size: int = 1000,
    query_size: int = 50,
    m_test: int = 100,
    epochs: int = 20,
    model_size: int = 1000,
    device: str = "cpu",
    save_results: bool = True,
    use_random_baseline: bool = False,
    output_dir: str = RESULTS_DIR,
    num_trials: int = 1
) -> AccuracyTestResults:
    """
    Run the complete accuracy test comparing influence scores with different sample sizes.
    Supports multiple trials with different random seeds for more robust results.
    """
    
    print("="*80)
    print("KRONFLUENCE TOY ACCURACY TEST")
    print("="*80)
    print(f"Dataset size: {dataset_size}")
    print(f"Query size: {query_size}")
    print(f"Model size target: ~{model_size} parameters")
    print(f"Number of trials: {num_trials}")
    
    if use_random_baseline:
        print(f"Baseline: M={m_test} vs Random Matrix")
        print(f"This tests correlation against random scores (should be ~0)")
    else:
        print(f"Sample sizes to test: M={m_test}, M={dataset_size} (full)")
    
    print(f"Training epochs: {epochs}")
    print(f"Device: {device}")
    print()
    
    # Run multiple trials
    trial_results = []
    base_seed = 42
    
    print(f"Running {num_trials} trial(s)...")
    
    for trial_num in range(1, num_trials + 1):
        trial_seed = base_seed + trial_num * 10000  # Ensure seeds are well separated
        
        trial_result = run_single_trial(
            dataset_size=dataset_size,
            query_size=query_size,
            m_test=m_test,
            epochs=epochs,
            model_size=model_size,
            device=device,
            use_random_baseline=use_random_baseline,
            trial_seed=trial_seed,
            trial_num=trial_num,
            total_trials=num_trials,
            output_dir=output_dir
        )
        
        trial_results.append(trial_result)
        print(f"    MSE: {trial_result['mse']:.8f}, Spearman: {trial_result['spearman_correlation']:.6f}")
    
    print()
    
    # Calculate averaged results
    mse_values = [r['mse'] for r in trial_results]
    spearman_values = [r['spearman_correlation'] for r in trial_results]
    spearman_p_values = [r['spearman_pvalue'] for r in trial_results]
    factor_time_test_values = [r['factor_time_test'] for r in trial_results]
    factor_time_full_values = [r['factor_time_full'] for r in trial_results]
    score_time_values = [r['score_time'] for r in trial_results]
    
    avg_mse = np.mean(mse_values)
    std_mse = np.std(mse_values) if num_trials > 1 else 0.0
    avg_spearman = np.mean(spearman_values)
    std_spearman = np.std(spearman_values) if num_trials > 1 else 0.0
    avg_spearman_p = np.mean(spearman_p_values)
    avg_factor_time_test = np.mean(factor_time_test_values)
    avg_factor_time_full = np.mean(factor_time_full_values)
    avg_score_time = np.mean(score_time_values)
    
    # Get model parameter count (should be same across trials)
    temp_model = ToyModel(input_dim=10, target_params=model_size)
    param_count = temp_model.count_parameters()
    
    # Create results
    results = AccuracyTestResults(
        model_params=param_count,
        model_size_target=model_size,
        dataset_size=dataset_size,
        query_size=query_size,
        epochs=epochs,
        m_test=m_test,
        m_full=dataset_size if not use_random_baseline else -1,
        mse_test_vs_full=avg_mse,
        spearman_correlation=avg_spearman,
        spearman_pvalue=avg_spearman_p,
        baseline_type="random" if use_random_baseline else "full_dataset",
        factor_time_test=avg_factor_time_test,
        factor_time_full=avg_factor_time_full,
        score_time=avg_score_time,
        timestamp=datetime.now().isoformat(),
        experiment_id=f"accuracy_test_m{m_test}_vs_{dataset_size if not use_random_baseline else 'random'}_{int(time.time())}",
        num_trials=num_trials,
        trial_results=trial_results,
        mse_std=std_mse,
        spearman_std=std_spearman
    )
    
    # Print results
    print("="*80)
    print("ACCURACY TEST RESULTS")
    print("="*80)
    print(f"Model parameters: {results.model_params:,}")
    print(f"Model size target: ~{results.model_size_target:,} parameters")
    print(f"Dataset size: {results.dataset_size:,}")
    print(f"Query size: {results.query_size:,}")
    print(f"Training epochs: {results.epochs}")
    print(f"Number of trials: {results.num_trials}")
    print()
    print("Comparison:")
    print(f"  Test M = {results.m_test}")
    print(f"  Baseline = {'random' if use_random_baseline else dataset_size}")
    print(f"  Baseline type = {results.baseline_type}")
    print()
    print("Accuracy Metrics (averaged across trials):")
    if num_trials > 1:
        print(f"  MSE: {results.mse_test_vs_full:.8f} ± {results.mse_std:.8f}")
        print(f"  Spearman Correlation: {results.spearman_correlation:.6f} ± {results.spearman_std:.6f} (p={results.spearman_pvalue:.2e})")
    else:
    print(f"  MSE: {results.mse_test_vs_full:.8f}")
    print(f"  Spearman Correlation: {results.spearman_correlation:.6f} (p={results.spearman_pvalue:.2e})")
    print()
    print("Interpretation:")
    if use_random_baseline:
        if abs(results.spearman_correlation) < 0.1:
            print("  ✅ Good - correlation with random is near zero as expected")
        else:
            print("  ⚠️  Warning - unexpectedly high correlation with random scores")
    else:
        if results.spearman_correlation > 0.9:
            print("  🥇 Excellent rank correlation - influence rankings well preserved")
        elif results.spearman_correlation > 0.7:
            print("  🥈 Good rank correlation - influence rankings reasonably preserved")
        elif results.spearman_correlation > 0.5:
            print("  🥉 Moderate rank correlation - some ranking preservation")
        else:
            print("  ❌ Poor rank correlation - ranking preservation is weak")
    print()
    print("Timing (averaged across trials):")
    print(f"  Factor computation time (M={results.m_test}): {results.factor_time_test:.2f}s") 
    if not use_random_baseline:
        print(f"  Factor computation time (M={dataset_size}): {results.factor_time_full:.2f}s")
        print(f"  Score computation time: {results.score_time:.2f}s")
    print()
    print("="*80)
    
    # Save results if requested
    if save_results:
        save_results_to_file(results, output_dir)
    
    return results


def save_results_to_file(results: AccuracyTestResults, output_dir: str) -> str:
    """Save results to a JSON file."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    filename = f"accuracy_test_m{results.m_test}_vs_{results.m_full}_{int(time.time())}.json"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w') as f:
        json.dump(asdict(results), f, indent=2)
    
    print(f"Results saved to: {filepath}")
    return filepath


def main():
    parser = argparse.ArgumentParser(
        description="Toy accuracy test for Kronfluence influence score computation"
    )
    
    parser.add_argument(
        "--dataset_size",
        type=int,
        default=1000,
        help="Number of training points (default: 1000)"
    )
    
    parser.add_argument(
        "--query_size", 
        type=int,
        default=50,
        help="Number of query points (default: 50)"
    )
    
    parser.add_argument(
        "--m_test",
        type=int,
        default=100,
        help="Test sample size K for factor computation (default: 100)"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs (default: 20)"
    )
    
    parser.add_argument(
        "--model_size",
        type=int,
        default=1000,
        choices=[1000, 2500, 5000, 7500, 10000, 12500, 15000, 17500, 20000],
        help="Target number of model parameters (default: 1000)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use for computation (default: cpu)"
    )
    
    parser.add_argument(
        "--random_baseline",
        action="store_true",
        help="Test correlation against random matrix instead of full dataset"
    )
    
    parser.add_argument(
        "--no_save",
        action="store_true",
        help="Don't save results to file"
    )
    
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default=RESULTS_DIR,
        help=f"Directory to save results (default: {RESULTS_DIR})"
    )
    
    parser.add_argument(
        "--num_trials",
        type=int,
        default=1,
        help="Number of trials to run with different random seeds (default: 1)"
    )
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Validate arguments
    if not args.random_baseline and args.m_test > args.dataset_size:
        raise ValueError(f"m_test ({args.m_test}) cannot be larger than dataset_size ({args.dataset_size})")
    
    if args.device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA requested but not available, falling back to CPU")
        args.device = "cpu"
    
    if args.num_trials < 1:
        raise ValueError(f"num_trials must be at least 1, got {args.num_trials}")
    
    # Run the accuracy test
    results = run_accuracy_test(
        dataset_size=args.dataset_size,
        query_size=args.query_size,
        m_test=args.m_test,
        epochs=args.epochs,
        model_size=args.model_size,
        device=args.device,
        save_results=not args.no_save,
        use_random_baseline=args.random_baseline,
        output_dir=args.output_dir,
        num_trials=args.num_trials
    )
    
    return results


if __name__ == "__main__":
    main()
