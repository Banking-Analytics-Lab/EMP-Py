# EMP-Py

EMP Python Package repository, currently at [version 2.0.4](https://pypi.org/project/EMP-PY/).


Functions for estimating EMP (Expected Maximum Profit Measure) in [Credit Risk Scoring](https://www.sciencedirect.com/science/article/pii/S0377221714003105) and [Customer Churn Prediction](https://ieeexplore.ieee.org/document/6165289), according to Verbraken et al (2013, 2014).

## Installation

```python
pip install EMP-PY
```

## Usage

```python
from EMP.metrics import empCreditScoring

scores = [0.34, 0.44, 0.67, 0.83]
classes = [0, 0, 1, 0]
k = 2

# By default, returns the output and does not print it (no rounding)
empCreditScoring(scores, classes)

# Prints and returns the output (no rounding)
empCreditScoring(scores, classes, print_output=True)

# Only prints the output (no rounding)
empCreditScoring(scores, classes, print_output=True, return_output=False)

# Returns the output with k decimal places
empCreditScoring(scores, classes, rounding=k)
```

The class labels use 1 for the event of interest: default in `empCreditScoring` and churn in `empChurn`. A higher score must mean that the event is more likely. EMP depends only on the order of the scores, so the scores do not have to be probabilities. Logits and scorecard points also work. Scorecard points usually increase with creditworthiness, so reverse their sign first.

The functions raise a `ValueError` if the input is not valid. For example, this happens if the labels have more than two classes or only one class, or if a score is NaN or infinite.

The functions have been co-authored by Thomas Verbraken, Seppe van den Brucke and [Cristián Bravo](https://github.com/CBravoR). Python translation by [Emiliano Peñaloza](https://emilianopp.com/#/home) and Alexander Hemming.
