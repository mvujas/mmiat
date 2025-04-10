# MMiat: Minimal Membership Inference Attack Toolkit

MMiat is a Python library for implementing and evaluating Membership Inference Attacks (MIAs) on machine learning models. It provides a simple and efficient way to test the privacy vulnerabilities of trained models by determining whether a given data point was used during training.

## Features

- Implementation of various membership inference attacks (currently including Loss Attack)
- Utilities for dataset partitioning and attack evaluation
- Metrics calculation for attack performance (AUC, accuracy)
- Visualization tools for comparing different attacks
- Reproducibility utilities for consistent results

## Installation

You can install MMiat directly from GitHub using pip:

```bash
pip install git+https://github.com/mvujas/mmiat
```

## Requirements

- Python 3.8 or higher

## Quick Start

The library provides a simple interface for implementing membership inference attacks. Here's a basic example:

```python
from mmiat.attacks import LossAttack
from mmiat.utils.data import AttackPredictionDataset
from mmiat.evaluation.aggregate_metrics import calculate_miametrics
from mmiat.evaluation.reporting.plot import create_mia_metric_report

# Create an attack instance
attack = LossAttack(device="cuda" if torch.cuda.is_available() else "cpu")

# Prepare your dataset for attack
attack_dataset = AttackPredictionDataset(trainset, nontrainset)

# Perform the attack
membership_labels, confidences = attack.attack(model, attack_dataset)

# Calculate metrics
metrics = calculate_miametrics(membership_labels, confidences)

# Visualize results
create_mia_metric_report(metrics, attack_name="Loss Attack", extras=["auc", "accuracy"])
```

## Examples

Check out the [examples](examples/) directory for more detailed usage examples:

- [Attacking a Simple Classifier](examples/attacking_simple_classifier.ipynb): Demonstrates how to perform a membership inference attack on a simple convolutional network trained on CIFAR-10.

## License

This project is licensed under the terms of the license included in the [LICENSE](LICENSE) file.
