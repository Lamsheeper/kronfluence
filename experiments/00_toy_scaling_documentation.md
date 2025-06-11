# Toy Model Scaling Tests for Kronfluence

This script provides an easy way to test how Kronfluence's influence function computations scale with different model sizes and dataset sizes using simple toy models.

## Features

- **Simple toy models**: MLPs with varying parameter counts (90 to 15K parameters)
- **Synthetic datasets**: Random feature vectors with structured targets
- **Automated scaling tests**: Test parameter scaling, data scaling, or single experiments
- **Timing measurements**: Separate timing for factor computation and score computation
- **Multiple strategies**: Support for identity, diagonal, kfac, and ekfac approximations

## Model Sizes

The script provides several predefined model architectures:

- **tiny**: ~90 parameters (single hidden layer with 8 units)
- **small**: ~400 parameters (two hidden layers: 16, 8)
- **medium**: ~1K parameters (two hidden layers: 32, 16)  
- **large**: ~4K parameters (three hidden layers: 64, 32, 16)
- **xlarge**: ~15K parameters (three hidden layers: 128, 64, 32)

## Usage Examples

### 1. Parameter Scaling Test
Test how computation time scales with model parameters (using fixed dataset size):

```bash
python toy_scaling_test.py --test params --dataset-size 100 --query-size 20
```

This will run experiments with tiny → small → medium → large models and show how timing scales.

### 2. Data Scaling Test  
Test how computation time scales with dataset size (using fixed model size):

```bash
python toy_scaling_test.py --test data --model-size small
```

This will test dataset sizes of 50, 100, 200, 500 samples with a small model.

### 3. Single Experiment
Run a single experiment with specific parameters:

```bash
python toy_scaling_test.py --test single --model-size medium --dataset-size 200 --query-size 50 --strategy ekfac
```

### 4. Test Different Strategies
Compare different Hessian approximation strategies:

```bash
# Test with diagonal approximation (faster)
python toy_scaling_test.py --test single --strategy diagonal

# Test with KFAC 
python toy_scaling_test.py --test single --strategy kfac

# Test with EKFAC (default, most accurate)
python toy_scaling_test.py --test single --strategy ekfac
```

## Expected Output

The script will show:
1. Model training progress
2. Factor computation time
3. Score computation time  
4. Final summary with parameter counts and timings

Example output:
```
=== Experiment: small model, 100 train samples, 20 query samples ===
Model parameters: 409
Training model...
Epoch 0, Loss: 0.8234
...
Computing factors...
Factor computation time: 2.15s
Computing influence scores...
Score computation time: 1.43s
Scores shape: torch.Size([20, 100])
Total time: 3.58s
```

## Understanding the Results

- **Factor time**: Time to compute the Hessian approximation factors
- **Score time**: Time to compute pairwise influence scores
- **Scores shape**: [query_size, train_size] - influence of each training point on each query point

## Tips for Scaling Studies

1. **Start small**: Begin with tiny models and small datasets to get baseline timings
2. **CPU vs GPU**: The script uses CPU by default for small models to avoid GPU overhead
3. **Memory limits**: Larger models/datasets may require adjusting batch sizes or using GPU
4. **Strategy comparison**: Try different strategies (diagonal is fastest, ekfac most accurate)
5. **Multiple runs**: Run multiple times for stable timing measurements

## Extending the Framework

You can easily modify the script to:
- Test different input dimensions
- Add more model architectures  
- Test with different data distributions
- Add classification tasks instead of regression
- Test with different noise levels

The framework provides a simple foundation for understanding Kronfluence's computational characteristics before applying it to your real models and datasets. 