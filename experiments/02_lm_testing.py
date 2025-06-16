#!/usr/bin/env python3
"""
Basic Language Model Influence Testing

This script tests influence function calculations on our toy language models.
It loads a trained model from the model-generator/models directory and computes
influence scores for a few query examples against the training data.

This is a basic test to verify that influence calculations work correctly
with our Hugging Face toy models.
"""

import argparse
import logging
import os
import time
import json
import tempfile
from typing import Tuple, Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset as HFDataset

from kronfluence.analyzer import Analyzer, prepare_model
from kronfluence.arguments import FactorArguments, ScoreArguments
from kronfluence.task import Task

# Set up more stable CUDA linear algebra backend
if torch.cuda.is_available():
    try:
        torch.backends.cuda.preferred_linalg_library("cusolver")
    except:
        try:
            torch.backends.cuda.preferred_linalg_library("magma")
        except:
            pass

BATCH_TYPE = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]  # input_ids, attention_mask, labels
RESULTS_DIR = "/share/u/yu.stev/influence/kronfluence/data/lm_testing"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class LMTestResults:
    """Store language model influence test results."""
    model_path: str
    model_params: int
    dataset_size: int
    query_size: int
    max_length: int
    sample_size: int  # Number of samples used for factor computation
    top_influences: List[Dict[str, Any]]  # Top influential examples for each query
    factor_time: float
    score_time: float
    timestamp: str
    experiment_id: str


class HFDatasetWrapper(Dataset):
    """Wrapper to make Hugging Face Dataset compatible with PyTorch DataLoader."""
    
    def __init__(self, hf_dataset: HFDataset, device: str = "cpu"):
        self.hf_dataset = hf_dataset
        self.device = device
        
    def __len__(self) -> int:
        return len(self.hf_dataset)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        example = self.hf_dataset[idx]
        # Move tensors to the specified device immediately
        input_ids = torch.tensor(example["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(example["attention_mask"], dtype=torch.long)
        labels = torch.tensor(example["labels"], dtype=torch.long)
        
        if self.device == "cuda":
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()
            labels = labels.cuda()
        
        return input_ids, attention_mask, labels


class LanguageModelTask(Task):
    """Task definition for language modeling with Hugging Face models."""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def compute_train_loss(
        self,
        batch: BATCH_TYPE,
        model: nn.Module,
        sample: bool = False,
    ) -> torch.Tensor:
        input_ids, attention_mask, labels = batch
        
        # Forward pass through the model
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        if not sample:
            return outputs.loss * input_ids.size(0)  # Scale by batch size for sum reduction
        
        # For sampling mode, we can use the same loss
        # (Hugging Face models already handle the loss computation properly)
        return outputs.loss * input_ids.size(0)
    
    def compute_measurement(
        self,
        batch: BATCH_TYPE,
        model: nn.Module,
    ) -> torch.Tensor:
        # Use same as training loss for language modeling
        return self.compute_train_loss(batch, model, sample=False)


def load_toy_model_and_data(model_path: str) -> Tuple[nn.Module, HFDataset, Dict]:
    """
    Load a trained toy language model and its training data.
    
    Args:
        model_path: Path to the model directory (e.g., "models/my_toy_model")
        
    Returns:
        Tuple of (model, training_dataset, metadata)
    """
    model_path = Path(model_path)
    
    # Load the Hugging Face model
    hf_model_path = model_path / "huggingface_model"
    if not hf_model_path.exists():
        raise FileNotFoundError(f"Hugging Face model not found at {hf_model_path}")
    
    logger.info(f"Loading model from {hf_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(hf_model_path)
    model = AutoModelForCausalLM.from_pretrained(hf_model_path)
    
    # Load training data
    training_data_dirs = [d for d in model_path.iterdir() 
                         if d.is_dir() and d.name.startswith("training_data")]
    
    if not training_data_dirs:
        raise FileNotFoundError(f"No training data found in {model_path}")
    
    training_data_path = training_data_dirs[0]
    logger.info(f"Loading training data from {training_data_path}")
    
    training_dataset = HFDataset.load_from_disk(str(training_data_path))
    
    # Load metadata
    metadata = {}
    metadata_path = training_data_path / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    
    logger.info(f"Loaded model with {sum(p.numel() for p in model.parameters()):,} parameters")
    logger.info(f"Loaded training dataset with {len(training_dataset)} examples")
    
    return model, training_dataset, metadata, tokenizer


def create_query_examples(tokenizer, num_queries: int = 5, max_length: int = 128) -> HFDataset:
    """
    Create a few query examples for influence testing.
    These are INCOMPLETE prompts that the model needs to complete,
    making influence testing more meaningful by focusing on predictions.
    """
    # Create incomplete prompts that match our new simple training pattern: [noun] [verb] [noun]
    # The model should predict the final word(s)
    query_prompts = [
        "cat finds",              # Should predict "ball", "toy", "food", etc.
        "dog sees",               # Should predict "ball", "stick", "toy", etc.
        "bird catches",           # Should predict "fish", "food", "mouse", etc.
        "mouse follows",          # Should predict "cat", "food", "stick", etc.
        "rabbit loves",           # Should predict "food", "toy", "treat", etc.
        "elephant carries",       # Should predict "stick", "toy", "food", etc.
        "fish follows",           # Should predict "food", "toy", "stick", etc.
        "horse runs",             # Should predict "fast", "away", "home", etc.
    ]
    
    # Take only the requested number of queries
    query_prompts = query_prompts[:num_queries]
    
    # For influence testing, we also need the "expected" completions
    # These help us understand what the model should predict
    expected_completions = [
        "cat finds ball",
        "dog sees toy",
        "bird catches fish",
        "mouse follows food",
        "rabbit loves treat",
        "elephant carries stick",
        "fish follows food",
        "horse runs fast",
    ]
    
    # Tokenize the queries
    tokenized_queries = []
    for i, prompt in enumerate(query_prompts):
        # Tokenize the incomplete prompt
        prompt_tokens = tokenizer(
            prompt,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors=None,
            add_special_tokens=False  # No special tokens for our simple format
        )
        
        # For the labels, we use the complete sentence
        # This allows us to compute loss on the completion part
        complete_text = expected_completions[i] if i < len(expected_completions) else prompt + " unknown"
        complete_tokens = tokenizer(
            complete_text,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors=None,
            add_special_tokens=False  # No special tokens for our simple format
        )
        
        # Create the example
        example = {
            "input_ids": complete_tokens["input_ids"],  # Full sequence for input
            "attention_mask": complete_tokens["attention_mask"],
            "labels": complete_tokens["input_ids"].copy(),  # Same as input_ids for causal LM
            "prompt_text": prompt,  # Store the incomplete prompt
            "complete_text": complete_text,  # Store the complete text
            "text": complete_text  # For compatibility with existing code
        }
        
        tokenized_queries.append(example)
    
    # Convert to HF Dataset
    query_dataset = HFDataset.from_list(tokenized_queries)
    
    logger.info(f"Created {len(query_dataset)} query examples:")
    for i, (prompt, complete) in enumerate(zip(query_prompts, expected_completions[:len(query_prompts)])):
        logger.info(f"  {i+1}: '{prompt}' → '{complete}'")
    
    return query_dataset


def compute_influence_scores(
    model: nn.Module,
    task: Task,
    train_dataset: HFDataset,
    query_dataset: HFDataset,
    sample_size: int,
    analysis_name: str,
    device: str = "cpu",
    output_dir: str = RESULTS_DIR
) -> Tuple[torch.Tensor, float, float]:
    """
    Compute influence scores for query examples against training data.
    
    Returns:
        Tuple of (influence_scores, factor_time, score_time)
    """
    
    # Set deterministic behavior for reproducibility (like the toy scripts)
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        # Ensure deterministic CUDA operations
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Create output directory
    analysis_output_dir = os.path.join(output_dir, "analysis", analysis_name)
    os.makedirs(analysis_output_dir, exist_ok=True)
    
    # Wrap datasets for PyTorch compatibility with proper device handling
    train_dataset_wrapped = HFDatasetWrapper(train_dataset, device=device)
    query_dataset_wrapped = HFDatasetWrapper(query_dataset, device=device)
    
    # Ensure model is on the correct device
    if device == "cuda":
        model = model.cuda()
        # Double-check all parameters are on CUDA
        for name, param in model.named_parameters():
            if not param.is_cuda:
                logger.warning(f"Parameter {name} not on CUDA, moving it")
                param.data = param.data.cuda()
    else:
        model = model.cpu()
        # Double-check all parameters are on CPU
        for name, param in model.named_parameters():
            if param.is_cuda:
                logger.warning(f"Parameter {name} on CUDA, moving to CPU")
                param.data = param.data.cpu()
    
    # Create analyzer with explicit device setting
    analyzer = Analyzer(
        analysis_name=analysis_name,
        model=model,
        task=task,
        cpu=(device == "cpu"),
        disable_tqdm=False,
        output_dir=analysis_output_dir
    )
    
    # Set up factor arguments with numerical stability (matching toy scripts)
    factor_args = FactorArguments(
        strategy="ekfac",
        covariance_max_examples=sample_size,
        lambda_max_examples=sample_size,
        # Numerical stability improvements from toy scripts
        eigendecomposition_dtype=torch.float64,  # Use double precision for eigendecomposition
        activation_covariance_dtype=torch.float32,
        gradient_covariance_dtype=torch.float32,
        # Add empirical Fisher for more stability (like toy scripts)
        use_empirical_fisher=True,  # Use actual labels instead of sampling for stability
    )
    
    # Compute factors with error handling
    logger.info("Computing influence factors...")
    factor_start_time = time.time()
    
    try:
        # Clear CUDA cache if using GPU
        if device == "cuda":
            torch.cuda.empty_cache()
            logger.info(f"CUDA memory before factor computation: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
        
        analyzer.fit_all_factors(
            factors_name="lm_factors",
            dataset=train_dataset_wrapped,
            per_device_batch_size=1 if device == "cuda" else 2,  # Smaller batch for GPU to avoid OOM
            factor_args=factor_args,
            overwrite_output_dir=True,
        )
        factor_time = time.time() - factor_start_time
        logger.info(f"Factor computation took {factor_time:.2f} seconds")
        
    except RuntimeError as e:
        if "CUDA" in str(e) or "out of memory" in str(e).lower():
            logger.error(f"CUDA error during factor computation: {e}")
            logger.info("Falling back to CPU computation...")
            
            # Clear CUDA cache and move everything to CPU
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Move model to CPU
            model = model.cpu()
            device = "cpu"
            
            # Recreate dataset wrappers for CPU
            train_dataset_wrapped = HFDatasetWrapper(train_dataset, device="cpu")
            query_dataset_wrapped = HFDatasetWrapper(query_dataset, device="cpu")
            
            # Recreate analyzer for CPU
            analyzer = Analyzer(
                analysis_name=analysis_name + "_cpu_fallback",
                model=model,
                task=task,
                cpu=True,
                disable_tqdm=False,
                output_dir=analysis_output_dir
            )
            
            try:
                analyzer.fit_all_factors(
                    factors_name="lm_factors",
                    dataset=train_dataset_wrapped,
                    per_device_batch_size=2,
                    factor_args=factor_args,
                    overwrite_output_dir=True,
                )
                factor_time = time.time() - factor_start_time
                logger.info(f"Factor computation (CPU fallback) took {factor_time:.2f} seconds")
                
            except Exception as e2:
                logger.error(f"CPU fallback also failed: {e2}")
                raise e2
        else:
            logger.error(f"Non-CUDA error during factor computation: {e}")
            raise e
    except Exception as e:
        logger.error(f"Unexpected error during factor computation: {e}")
        raise e
    
    # Compute pairwise scores
    logger.info("Computing influence scores...")
    score_start_time = time.time()
    
    try:
        # Clear CUDA cache if using GPU
        if device == "cuda":
            torch.cuda.empty_cache()
            logger.info(f"CUDA memory before score computation: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
        
        analyzer.compute_pairwise_scores(
            scores_name="lm_scores",
            factors_name="lm_factors",
            query_dataset=query_dataset_wrapped,
            train_dataset=train_dataset_wrapped,
            per_device_query_batch_size=1,  # Very small batch size for stability
            per_device_train_batch_size=1 if device == "cuda" else 2,  # Even smaller for GPU
            overwrite_output_dir=True,
        )
        score_time = time.time() - score_start_time
        logger.info(f"Score computation took {score_time:.2f} seconds")
        
    except RuntimeError as e:
        if "CUDA" in str(e) or "out of memory" in str(e).lower():
            logger.error(f"CUDA error during score computation: {e}")
            # If we're already on CPU, this is a different issue
            if device == "cpu":
                logger.error("Score computation failed even on CPU")
                raise e
            else:
                logger.error("Score computation failed on GPU, but factors were computed on GPU")
                logger.error("Cannot fallback to CPU for scores when factors are on GPU")
                raise e
        else:
            logger.error(f"Non-CUDA error during score computation: {e}")
            raise e
    except Exception as e:
        logger.error(f"Unexpected error during score computation: {e}")
        raise e
    
    # Load the computed scores
    scores = analyzer.load_pairwise_scores("lm_scores")["all_modules"]
    
    return scores, factor_time, score_time


def analyze_top_influences(
    scores: torch.Tensor,
    train_dataset: HFDataset,
    query_dataset: HFDataset,
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    Analyze the top influential training examples for each query.
    
    Args:
        scores: Influence scores tensor of shape (num_queries, num_train_examples)
        train_dataset: Training dataset
        query_dataset: Query dataset
        top_k: Number of top influences to return per query
        
    Returns:
        List of dictionaries with top influences for each query
    """
    results = []
    
    for query_idx in range(scores.shape[0]):
        query_text = query_dataset[query_idx]["text"]
        query_scores = scores[query_idx]  # Shape: (num_train_examples,)
        
        # Get top-k most influential (highest scores)
        top_indices = torch.topk(query_scores, k=top_k, largest=True).indices
        top_scores = query_scores[top_indices]
        
        # Get bottom-k least influential (lowest/most negative scores)
        bottom_indices = torch.topk(query_scores, k=top_k, largest=False).indices
        bottom_scores = query_scores[bottom_indices]
        
        query_result = {
            "query_idx": query_idx,
            "query_text": query_text,
            "top_positive_influences": [],
            "top_negative_influences": []
        }
        
        # Top positive influences
        for i, (idx, score) in enumerate(zip(top_indices, top_scores)):
            train_text = train_dataset[idx.item()]["text"]
            query_result["top_positive_influences"].append({
                "rank": i + 1,
                "train_idx": idx.item(),
                "train_text": train_text,
                "influence_score": score.item()
            })
        
        # Top negative influences
        for i, (idx, score) in enumerate(zip(bottom_indices, bottom_scores)):
            train_text = train_dataset[idx.item()]["text"]
            query_result["top_negative_influences"].append({
                "rank": i + 1,
                "train_idx": idx.item(),
                "train_text": train_text,
                "influence_score": score.item()
            })
        
        results.append(query_result)
    
    return results


def generate_predictions(
    model: nn.Module,
    tokenizer,
    query_dataset: HFDataset,
    max_new_tokens: int = 10,
    device: str = "cpu"
) -> List[Dict[str, Any]]:
    """
    Generate predictions from the model on query prompts.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        query_dataset: Dataset with query examples
        max_new_tokens: Maximum number of tokens to generate
        device: Device to run on
        
    Returns:
        List of prediction results
    """
    model.eval()
    predictions = []
    
    logger.info("Generating model predictions on query prompts...")
    
    with torch.no_grad():
        for i, example in enumerate(query_dataset):
            # Get the prompt (incomplete text)
            prompt_text = example.get("prompt_text", example["text"])
            expected_text = example.get("complete_text", example["text"])
            
            # Tokenize just the prompt
            prompt_tokens = tokenizer(
                prompt_text,
                return_tensors="pt",
                truncation=True,
                max_length=128
            )
            
            # Move to device
            if device == "cuda":
                prompt_tokens = {k: v.cuda() for k, v in prompt_tokens.items()}
            
            # Generate completion
            try:
                with torch.no_grad():
                    generated = model.generate(
                        input_ids=prompt_tokens["input_ids"],
                        attention_mask=prompt_tokens["attention_mask"],
                        max_new_tokens=max_new_tokens,
                        do_sample=False,  # Deterministic generation
                        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0,
                        temperature=1.0,
                        top_p=1.0
                    )
                
                # Decode the generated text
                generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
                
                # Extract just the completion (remove the prompt part)
                completion = generated_text[len(prompt_text):].strip()
                
                prediction_result = {
                    "query_idx": i,
                    "prompt": prompt_text,
                    "expected": expected_text,
                    "generated": generated_text,
                    "completion": completion,
                    "matches_expected": completion.lower() in expected_text.lower()
                }
                
                predictions.append(prediction_result)
                
                logger.info(f"Query {i+1}: '{prompt_text}' → '{completion}'")
                
            except Exception as e:
                logger.warning(f"Failed to generate for query {i}: {e}")
                predictions.append({
                    "query_idx": i,
                    "prompt": prompt_text,
                    "expected": expected_text,
                    "generated": "[GENERATION FAILED]",
                    "completion": "[FAILED]",
                    "matches_expected": False
                })
    
    return predictions


def print_predictions_and_influences(
    predictions: List[Dict[str, Any]],
    top_influences: List[Dict[str, Any]]
):
    """Print model predictions alongside influence analysis."""
    
    logger.info("\n" + "="*80)
    logger.info("MODEL PREDICTIONS & INFLUENCE ANALYSIS")
    logger.info("="*80)
    
    for pred, inf in zip(predictions, top_influences):
        query_idx = pred["query_idx"]
        
        logger.info(f"\nQuery {query_idx + 1}:")
        logger.info(f"  Prompt:    '{pred['prompt']}'")
        logger.info(f"  Expected:  '{pred['expected']}'")
        logger.info(f"  Generated: '{pred['generated']}'")
        logger.info(f"  Completion: '{pred['completion']}'")
        logger.info(f"  Correct: {'✓' if pred['matches_expected'] else '✗'}")
        logger.info("-" * 60)
        
        logger.info("Training examples that HELPED this prediction:")
        for influence in inf["top_positive_influences"][:3]:  # Show top 3
            logger.info(f"  +{influence['influence_score']:6.3f} | {influence['train_text']}")
        
        logger.info("Training examples that HURT this prediction:")
        for influence in inf["top_negative_influences"][:3]:  # Show top 3
            logger.info(f"  {influence['influence_score']:7.3f} | {influence['train_text']}")
        
        logger.info("")


def run_basic_lm_test(
    model_path: str,
    num_queries: int = 5,
    sample_size: Optional[int] = None,
    device: str = "cpu",
    output_dir: str = RESULTS_DIR
) -> LMTestResults:
    """
    Run a basic influence function test on a toy language model.
    
    Args:
        model_path: Path to the trained model directory
        num_queries: Number of query examples to test
        sample_size: Number of training examples to use for factor computation (None = use all, but capped at 500 for stability)
        device: Device to run on ("cpu" or "cuda")
        output_dir: Directory to save results
        
    Returns:
        Test results
    """
    
    logger.info("="*80)
    logger.info("BASIC LANGUAGE MODEL INFLUENCE TEST")
    logger.info("="*80)
    logger.info(f"Model path: {model_path}")
    logger.info(f"Number of queries: {num_queries}")
    logger.info(f"Device: {device}")
    
    # Load model and data
    model, train_dataset, metadata, tokenizer = load_toy_model_and_data(model_path)
    
    # Determine sample size with stability limits
    if sample_size is None:
        # Use all data but cap at 500 for numerical stability
        sample_size = min(len(train_dataset), 500)
        logger.info(f"Auto-selected sample size: {sample_size} (capped for stability)")
    else:
        sample_size = min(sample_size, len(train_dataset))
        logger.info(f"Using requested sample size: {sample_size}")
    
    # Additional stability check
    if sample_size > 1000:
        logger.warning(f"Large sample size ({sample_size}) may cause numerical instability. Consider using --sample_size 500 or less.")
    
    # Create query examples
    max_length = metadata.get("max_length", 128)
    query_dataset = create_query_examples(tokenizer, num_queries, max_length)
    
    # Prepare model and task
    task = LanguageModelTask(tokenizer)
    
    # Move model to device BEFORE preparing and ensure it stays there
    if device == "cuda" and torch.cuda.is_available():
        model = model.cuda()
        logger.info("Moved model to CUDA")
        # Verify model is on CUDA
        for param in model.parameters():
            if not param.is_cuda:
                logger.warning("Found parameter not on CUDA, moving it")
                param.data = param.data.cuda()
    else:
        model = model.cpu()
        device = "cpu"
        logger.info("Using CPU device")
        # Verify model is on CPU
        for param in model.parameters():
            if param.is_cuda:
                logger.warning("Found parameter on CUDA, moving to CPU")
                param.data = param.data.cpu()
    
    model = prepare_model(model, task)
    
    # Set deterministic behavior for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Compute influence scores
    analysis_name = f"lm_test_{int(time.time())}"
    scores, factor_time, score_time = compute_influence_scores(
        model, task, train_dataset, query_dataset, sample_size, analysis_name, device, output_dir
    )
    
    logger.info(f"Computed influence scores with shape: {scores.shape}")
    
    # Check for NaN or infinite values in scores
    if torch.isnan(scores).any():
        logger.warning("Found NaN values in influence scores!")
    if torch.isinf(scores).any():
        logger.warning("Found infinite values in influence scores!")
    
    # Analyze top influences
    logger.info("Analyzing top influential examples...")
    top_influences = analyze_top_influences(scores, train_dataset, query_dataset, top_k=5)
    
    # Generate predictions
    predictions = generate_predictions(model, tokenizer, query_dataset, max_new_tokens=10, device=device)
    
    # Print predictions and influences
    print_predictions_and_influences(predictions, top_influences)
    
    # Create results object
    results = LMTestResults(
        model_path=model_path,
        model_params=sum(p.numel() for p in model.parameters()),
        dataset_size=len(train_dataset),
        query_size=len(query_dataset),
        max_length=max_length,
        sample_size=sample_size,
        top_influences=top_influences,
        factor_time=factor_time,
        score_time=score_time,
        timestamp=datetime.now().isoformat(),
        experiment_id=analysis_name
    )
    
    # Save results
    save_results(results, output_dir)
    
    return results


def print_influence_results(top_influences: List[Dict[str, Any]]):
    """Print the influence analysis results in a readable format."""
    
    logger.info("\n" + "="*80)
    logger.info("INFLUENCE ANALYSIS RESULTS")
    logger.info("="*80)
    
    for query_result in top_influences:
        # Extract prompt and completion info if available
        query_text = query_result['query_text']
        query_idx = query_result['query_idx']
        
        logger.info(f"\nQuery {query_idx + 1}: '{query_text}'")
        logger.info("  (Model was asked to complete this text)")
        logger.info("-" * 60)
        
        logger.info("Top POSITIVE influences (training examples that HELP the model predict correctly):")
        for inf in query_result["top_positive_influences"]:
            logger.info(f"  {inf['rank']}. [{inf['train_idx']:4d}] {inf['influence_score']:+8.4f} | {inf['train_text']}")
        
        logger.info("Top NEGATIVE influences (training examples that HURT the model's prediction):")
        for inf in query_result["top_negative_influences"]:
            logger.info(f"  {inf['rank']}. [{inf['train_idx']:4d}] {inf['influence_score']:+8.4f} | {inf['train_text']}")
        
        logger.info("")


def save_results(results: LMTestResults, output_dir: str) -> str:
    """Save results to a JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    
    filename = f"lm_test_results_{results.experiment_id}.json"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w') as f:
        json.dump(asdict(results), f, indent=2)
    
    logger.info(f"Results saved to {filepath}")
    return filepath


def main():
    parser = argparse.ArgumentParser(description="Test influence functions on toy language models")
    parser.add_argument("model_path", type=str, help="Path to the trained model directory")
    parser.add_argument("--num_queries", type=int, default=5, help="Number of query examples to test")
    parser.add_argument("--sample_size", type=int, help="Number of training examples for factor computation (default: auto-select, capped at 500 for stability)")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"], help="Device to run on")
    parser.add_argument("--output_dir", type=str, default=RESULTS_DIR, help="Output directory for results")
    
    args = parser.parse_args()
    
    # Validate model path exists
    if not os.path.exists(args.model_path):
        logger.error(f"Model path does not exist: {args.model_path}")
        return
    
    # Check CUDA availability
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available. Falling back to CPU.")
        args.device = "cpu"
    
    try:
        results = run_basic_lm_test(
            model_path=args.model_path,
            num_queries=args.num_queries,
            sample_size=args.sample_size,
            device=args.device,
            output_dir=args.output_dir
        )
        
        logger.info("\n" + "="*80)
        logger.info("TEST COMPLETED SUCCESSFULLY!")
        logger.info("="*80)
        logger.info(f"Total time: {results.factor_time + results.score_time:.2f} seconds")
        logger.info(f"  - Factor computation: {results.factor_time:.2f} seconds")
        logger.info(f"  - Score computation: {results.score_time:.2f} seconds")
        logger.info(f"Results saved to: {RESULTS_DIR}")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        logger.info("\nTroubleshooting tips:")
        logger.info("1. Try with --device cpu for more stability")
        logger.info("2. Use --sample_size 100 for faster testing")
        logger.info("3. Make sure the model path contains both 'huggingface_model/' and 'training_data_*/' directories")
        raise


if __name__ == "__main__":
    main()
