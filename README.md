# EMP-Py

EMP Python Package repository, currently at version 3.8.5.

Functions for estimating EMP (Expected Maximum Profit Measure) in Credit Risk Scoring and Customer Churn Prediction, according to Verbraken et al (2013, 2014).

## Installation

```python
pip install EMP
```

## Usage

```python
from EMP import empCreditScoring

scores = [0.34, 0.44, 0.67, 0.83]
classes = [0, 0, 1, 0]
k = 2

# By default will print and return output (no rounding)
empCreditScoring(scores, classes)

# Will only return output (no rounding)
empCreditScoring(scores, classes, print_output=False)

# Will only print output (no rounding)
empCreditScoring(scores, classes, return_output=False)

# Will print and return output with k decimal points
empCreditScoring(scores, classes, rounding=k)
```

The functions have been coathored by Thomas Verbraken, Seppe van den Brucke and Cristián Bravo.
