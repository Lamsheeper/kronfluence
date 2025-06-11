"""
Fisher/Hessian Approximation Accuracy Test

This test measures the accuracy of our Fisher/Hessian approximation methods
(KFAC, EKFAC, Diagonal) by comparing them against ground truth computations.
The accuracy is measured using L2 norm of the difference between approximation
and ground truth.

Model sizes: 10^3 to 10^4 parameters
Training data: 10^3 to 10^4 samples
"""

import math
import time
import os
import tempfile
import shutil
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from kronfluence.arguments import FactorArguments
from kronfluence.analyzer import Analyzer
from kronfluence.factor.config import FactorStrategy
from kronfluence.state import State
from kronfluence.task import Task
from kronfluence.utils.model import apply_ddp, get_tracked_module_names
from kronfluence.utils.constants import (
    ACTIVATION_EIGENVECTORS_NAME,
    ACTIVATION_EIGENVALUES_NAME,
    GRADIENT_EIGENVECTORS_NAME,
    GRADIENT_EIGENVALUES_NAME,
    LAMBDA_MATRIX_NAME,
)


class ToyRegressionTask(Task):
    """Simple regression task for testing."""
    
    def compute_train_loss(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        model: nn.Module,
        sample: bool = False,
    ) -> torch.Tensor:
        inputs, targets = batch
        outputs = model(inputs)
        if not sample:
            return F.mse_loss(outputs, targets, reduction="sum")
        # Sample outputs for true Fisher computation
        with torch.no_grad():
            sampled_targets = torch.normal(outputs.detach(), std=math.sqrt(0.1))
        return F.mse_loss(outputs, sampled_targets, reduction="sum")

    def compute_measurement(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        model: nn.Module,
    ) -> torch.Tensor:
        return self.compute_train_loss(batch, model, sample=False)

    def tracked_modules(self) -> Optional[List[str]]:
        return None  # Track all modules


class ToyClassificationTask(Task):
    """Simple classification task for testing."""
    
    def compute_train_loss(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        model: nn.Module,
        sample: bool = False,
    ) -> torch.Tensor:
        inputs, targets = batch
        logits = model(inputs)
        if not sample:
            return F.cross_entropy(logits, targets, reduction="sum")
        # Sample from model predictions for true Fisher
        with torch.no_grad():
            probs = F.softmax(logits.detach(), dim=-1)
            sampled_targets = torch.multinomial(probs, num_samples=1).squeeze()
        return F.cross_entropy(logits, sampled_targets, reduction="sum")

    def compute_measurement(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        model: nn.Module,
    ) -> torch.Tensor:
        return self.compute_train_loss(batch, model, sample=False)

    def tracked_modules(self) -> Optional[List[str]]:
        return None


def create_toy_regression_model(input_dim: int, hidden_dims: List[int], output_dim: int) -> nn.Module:
    """Create a small MLP for regression."""
    layers = []
    prev_dim = input_dim
    
    for i, hidden_dim in enumerate(hidden_dims):
        layers.append(nn.Linear(prev_dim, hidden_dim))
        if i < len(hidden_dims) - 1:  # No activation after last hidden layer
            layers.append(nn.ReLU())
        prev_dim = hidden_dim
    
    layers.append(nn.Linear(prev_dim, output_dim))
    return nn.Sequential(*layers)


def create_toy_classification_model(input_dim: int, hidden_dims: List[int], num_classes: int) -> nn.Module:
    """Create a small MLP for classification."""
    layers = []
    prev_dim = input_dim
    
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, hidden_dim))
        layers.append(nn.ReLU())
        prev_dim = hidden_dim
    
    layers.append(nn.Linear(prev_dim, num_classes))
    return nn.Sequential(*layers)


def generate_regression_data(n_samples: int, input_dim: int, output_dim: int, 
                           noise_std: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate synthetic regression data."""
    X = torch.randn(n_samples, input_dim)
    # Create some ground truth relationship
    true_weights = torch.randn(input_dim, output_dim)
    y = X @ true_weights + noise_std * torch.randn(n_samples, output_dim)
    return X, y


def generate_classification_data(n_samples: int, input_dim: int, num_classes: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate synthetic classification data."""
    X = torch.randn(n_samples, input_dim)
    # Create some ground truth classification boundary
    weights = torch.randn(input_dim, num_classes)
    logits = X @ weights
    y = torch.argmax(logits, dim=1)
    return X, y


def compute_exact_fisher_matrix(model: nn.Module, dataloader: DataLoader, task: Task) -> torch.Tensor:
    """
    Compute the exact Fisher Information Matrix for comparison.
    This is computationally expensive and only feasible for small models.
    """
    # Get all parameters
    params = []
    for param in model.parameters():
        if param.requires_grad:
            params.append(param.view(-1))
    total_params = sum(p.numel() for p in params)
    
    # Initialize Fisher matrix
    fisher_matrix = torch.zeros(total_params, total_params, device=next(model.parameters()).device)
    
    model.eval()
    total_samples = 0
    
    for batch in dataloader:
        inputs, targets = batch
        batch_size = inputs.size(0)
        
        for i in range(batch_size):
            # Single sample
            single_input = inputs[i:i+1]
            single_target = targets[i:i+1]
            
            # Forward pass
            model.zero_grad()
            loss = task.compute_train_loss((single_input, single_target), model, sample=True)
            
            # Backward pass to get gradients
            loss.backward()
            
            # Extract gradients
            grads = []
            for param in model.parameters():
                if param.requires_grad and param.grad is not None:
                    grads.append(param.grad.view(-1))
            
            if grads:
                grad_vector = torch.cat(grads)
                # Add outer product to Fisher matrix
                fisher_matrix += torch.outer(grad_vector, grad_vector)
        
        total_samples += batch_size
    
    # Average over samples
    fisher_matrix /= total_samples
    return fisher_matrix


def reconstruct_kfac_fisher_matrix(eigendecomposition_factors, lambda_factors, module_names, device):
    """
    Reconstruct the full Fisher matrix from KFAC factors.
    For KFAC: F^{-1} ≈ (A ⊗ G)^{-1} where A and G are covariance matrices
    """
    block_matrices = []
    
    for module_name in module_names:
        # Get eigenvectors and eigenvalues
        A_eigvecs = eigendecomposition_factors[ACTIVATION_EIGENVECTORS_NAME][module_name].to(device)
        G_eigvecs = eigendecomposition_factors[GRADIENT_EIGENVECTORS_NAME][module_name].to(device)
        A_eigvals = eigendecomposition_factors[ACTIVATION_EIGENVALUES_NAME][module_name].to(device)
        G_eigvals = eigendecomposition_factors[GRADIENT_EIGENVALUES_NAME][module_name].to(device)
        
        # Reconstruct covariance matrices
        A_cov = A_eigvecs @ torch.diag(A_eigvals) @ A_eigvecs.t()
        G_cov = G_eigvecs @ torch.diag(G_eigvals) @ G_eigvecs.t()
        
        # Compute Kronecker product approximation
        # F ≈ G ⊗ A (note the order)
        fisher_block = torch.kron(G_cov, A_cov)
        block_matrices.append(fisher_block)
    
    # Concatenate all blocks to form full matrix
    if len(block_matrices) == 1:
        return block_matrices[0]
    else:
        # Block diagonal matrix
        total_size = sum(mat.size(0) for mat in block_matrices)
        full_matrix = torch.zeros(total_size, total_size, device=device)
        start_idx = 0
        for mat in block_matrices:
            end_idx = start_idx + mat.size(0)
            full_matrix[start_idx:end_idx, start_idx:end_idx] = mat
            start_idx = end_idx
        return full_matrix


def reconstruct_ekfac_fisher_matrix(eigendecomposition_factors, lambda_factors, module_names, device):
    """
    Reconstruct the full Fisher matrix from EKFAC factors.
    EKFAC uses corrected eigenvalues (Lambda matrices)
    """
    block_matrices = []
    
    for module_name in module_names:
        # Get eigenvectors and lambda matrix
        A_eigvecs = eigendecomposition_factors[ACTIVATION_EIGENVECTORS_NAME][module_name].to(device)
        G_eigvecs = eigendecomposition_factors[GRADIENT_EIGENVECTORS_NAME][module_name].to(device)
        lambda_matrix = lambda_factors[LAMBDA_MATRIX_NAME][module_name].to(device)
        
        # EKFAC reconstruction: F ≈ (G ⊗ A) where eigenvalues are corrected
        # The lambda matrix contains the corrected eigenvalues
        # We need to reconstruct the approximation
        
        # Create the Fisher approximation using corrected eigenvalues
        # This is a simplified reconstruction - full EKFAC is more complex
        fisher_block = torch.kron(G_eigvecs @ torch.diag(lambda_matrix.mean(dim=0)) @ G_eigvecs.t(),
                                 A_eigvecs @ torch.diag(lambda_matrix.mean(dim=1)) @ A_eigvecs.t())
        block_matrices.append(fisher_block)
    
    # Concatenate all blocks
    if len(block_matrices) == 1:
        return block_matrices[0]
    else:
        total_size = sum(mat.size(0) for mat in block_matrices)
        full_matrix = torch.zeros(total_size, total_size, device=device)
        start_idx = 0
        for mat in block_matrices:
            end_idx = start_idx + mat.size(0)
            full_matrix[start_idx:end_idx, start_idx:end_idx] = mat
            start_idx = end_idx
        return full_matrix


def reconstruct_diagonal_fisher_matrix(lambda_factors, module_names, device):
    """
    Reconstruct the diagonal Fisher matrix approximation.
    """
    diagonal_elements = []
    
    for module_name in module_names:
        lambda_matrix = lambda_factors[LAMBDA_MATRIX_NAME][module_name].to(device)
        # For diagonal approximation, lambda matrix is diagonal
        diagonal_elements.append(lambda_matrix.view(-1))
    
    # Concatenate all diagonal elements
    full_diagonal = torch.cat(diagonal_elements)
    return torch.diag(full_diagonal)


def compute_approximation_fisher_matrix(model: nn.Module, dataloader: DataLoader, 
                                      task: Task, strategy: str, state: State) -> torch.Tensor:
    """
    Compute Fisher matrix approximation using the specified strategy.
    """
    # Create temporary directory for factors
    temp_dir = tempfile.mkdtemp()
    
    try:
        factor_args = FactorArguments(
            strategy=strategy,
            use_empirical_fisher=False,  # Use true Fisher
            covariance_max_examples=None,  # Use all data
            lambda_max_examples=None,
        )
        
        # Initialize analyzer
        analyzer = Analyzer(
            analysis_name="fisher_test",
            model=model,
            task=task,
            state=state,
            cache_dir=temp_dir,
            logging_level="ERROR"  # Reduce logging output
        )
        
        # Fit all factors
        analyzer.fit_all_factors(
            factors_name="test_factors",
            dataset=dataloader.dataset,
            factor_args=factor_args,
            per_device_batch_size=32,
            overwrite_output_dir=True,
        )
        
        # Load the computed factors
        device = next(model.parameters()).device
        module_names = get_tracked_module_names(model)
        
        if strategy == "diagonal":
            lambda_factors = analyzer.load_lambda_matrices(factors_name="test_factors")
            return reconstruct_diagonal_fisher_matrix(lambda_factors, module_names, device)
        
        elif strategy in ["kfac", "ekfac"]:
            eigendecomposition_factors = analyzer.load_eigendecomposition(factors_name="test_factors")
            
            if strategy == "kfac":
                return reconstruct_kfac_fisher_matrix(
                    eigendecomposition_factors, None, module_names, device
                )
            else:  # ekfac
                lambda_factors = analyzer.load_lambda_matrices(factors_name="test_factors")
                return reconstruct_ekfac_fisher_matrix(
                    eigendecomposition_factors, lambda_factors, module_names, device
                )
        
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
            
    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)


def run_accuracy_test(model_config: Dict, data_config: Dict, task_type: str = "regression") -> Dict:
    """
    Run accuracy test for a given model and data configuration.
    
    Args:
        model_config: Dict with model parameters
        data_config: Dict with data generation parameters  
        task_type: "regression" or "classification"
    
    Returns:
        Dict with test results including L2 norms
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model
    if task_type == "regression":
        model = create_toy_regression_model(
            input_dim=model_config["input_dim"],
            hidden_dims=model_config["hidden_dims"],
            output_dim=model_config["output_dim"]
        ).to(device)
        task = ToyRegressionTask()
        
        # Generate data
        X, y = generate_regression_data(
            n_samples=data_config["n_samples"],
            input_dim=model_config["input_dim"],
            output_dim=model_config["output_dim"],
            noise_std=data_config.get("noise_std", 0.1)
        )
    else:
        model = create_toy_classification_model(
            input_dim=model_config["input_dim"],
            hidden_dims=model_config["hidden_dims"],
            num_classes=model_config["num_classes"]
        ).to(device)
        task = ToyClassificationTask()
        
        # Generate data
        X, y = generate_classification_data(
            n_samples=data_config["n_samples"],
            input_dim=model_config["input_dim"],
            num_classes=model_config["num_classes"]
        )
    
    # Create dataloader
    dataset = TensorDataset(X.to(device), y.to(device))
    dataloader = DataLoader(dataset, batch_size=data_config.get("batch_size", 32), shuffle=False)
    
    # Prepare model for kronfluence
    model = apply_ddp(model, State())
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model has {total_params} parameters")
    print(f"Data has {data_config['n_samples']} samples")
    
    # Skip if model is too large for exact computation
    if total_params > 10000:
        print("Model too large for exact Fisher computation, skipping...")
        return {"error": "Model too large for exact computation"}
    
    # Compute ground truth Fisher matrix
    print("Computing exact Fisher matrix...")
    start_time = time.time()
    try:
        exact_fisher = compute_exact_fisher_matrix(model, dataloader, task)
        exact_time = time.time() - start_time
        print(f"Exact Fisher computation took {exact_time:.2f} seconds")
    except Exception as e:
        print(f"Failed to compute exact Fisher: {e}")
        return {"error": str(e)}
    
    results = {
        "model_params": total_params,
        "data_samples": data_config["n_samples"],
        "exact_fisher_time": exact_time,
        "approximations": {}
    }
    
    # Test each approximation strategy
    strategies = ["diagonal", "kfac", "ekfac"]
    state = State()
    
    for strategy in strategies:
        print(f"\nTesting {strategy.upper()} approximation...")
        start_time = time.time()
        
        try:
            approx_fisher = compute_approximation_fisher_matrix(
                model, dataloader, task, strategy, state
            )
            approx_time = time.time() - start_time
            
            # Compute L2 norm of difference
            diff = exact_fisher - approx_fisher
            l2_norm = torch.norm(diff, p=2).item()
            relative_error = l2_norm / torch.norm(exact_fisher, p=2).item()
            
            results["approximations"][strategy] = {
                "l2_norm_diff": l2_norm,
                "relative_error": relative_error,
                "computation_time": approx_time,
                "speedup": exact_time / approx_time if approx_time > 0 else float('inf')
            }
            
            print(f"{strategy.upper()} - L2 norm difference: {l2_norm:.6f}")
            print(f"{strategy.upper()} - Relative error: {relative_error:.6f}")
            print(f"{strategy.upper()} - Computation time: {approx_time:.2f}s")
            print(f"{strategy.upper()} - Speedup: {results['approximations'][strategy]['speedup']:.2f}x")
            
        except Exception as e:
            print(f"Failed to compute {strategy} approximation: {e}")
            results["approximations"][strategy] = {"error": str(e)}
    
    return results


def main():
    """Run the accuracy tests with different model and data sizes."""
    
    # Test configurations - starting with very small models
    test_configs = [
        {
            "name": "Tiny Model, Small Data",
            "model": {
                "input_dim": 5,
                "hidden_dims": [10],
                "output_dim": 1,
                "num_classes": 2  # for classification
            },
            "data": {
                "n_samples": 50,
                "batch_size": 16
            }
        },
        {
            "name": "Small Model, Small Data",
            "model": {
                "input_dim": 10,
                "hidden_dims": [20, 10],
                "output_dim": 1,
                "num_classes": 3  # for classification
            },
            "data": {
                "n_samples": 100,
                "batch_size": 16
            }
        },
        {
            "name": "Medium Model, Medium Data",
            "model": {
                "input_dim": 20,
                "hidden_dims": [40, 20],
                "output_dim": 1,
                "num_classes": 5
            },
            "data": {
                "n_samples": 500,
                "batch_size": 32
            }
        }
    ]
    
    all_results = {}
    
    for config in test_configs:
        print(f"\n{'='*60}")
        print(f"Running test: {config['name']}")
        print(f"{'='*60}")
        
        # Test regression
        print(f"\n--- REGRESSION TEST ---")
        reg_results = run_accuracy_test(
            model_config=config["model"],
            data_config=config["data"],
            task_type="regression"
        )
        
        # Test classification  
        print(f"\n--- CLASSIFICATION TEST ---")
        clf_results = run_accuracy_test(
            model_config=config["model"],
            data_config=config["data"],
            task_type="classification"
        )
        
        all_results[config["name"]] = {
            "regression": reg_results,
            "classification": clf_results
        }
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY OF RESULTS")
    print(f"{'='*60}")
    
    for test_name, results in all_results.items():
        print(f"\n{test_name}:")
        for task_type, task_results in results.items():
            if "error" not in task_results:
                print(f"  {task_type.title()}:")
                print(f"    Model params: {task_results['model_params']}")
                print(f"    Data samples: {task_results['data_samples']}")
                for strategy, metrics in task_results.get("approximations", {}).items():
                    if "error" not in metrics:
                        print(f"    {strategy.upper()}: L2={metrics['l2_norm_diff']:.6f}, "
                              f"RelErr={metrics['relative_error']:.6f}, "
                              f"Speedup={metrics['speedup']:.2f}x")
            else:
                print(f"  {task_type.title()}: {task_results['error']}")


if __name__ == "__main__":
    main()
