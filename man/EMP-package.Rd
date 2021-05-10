\name{EMP-package}
\alias{EMP-package}
\alias{EMP}
\docType{package}
\title{
Expected Maximum Profit Classification Performance Measure
}
\description{
The EMP measure is an alternative to AUC that includes the expected profit of
a given model, when compared to a baseline (no model used). Presented in 
Verbraken et al. (2014) as a preferred measure for credit risk scoring in any 
profit-driven environment and in Verbraken et al. (2013) as a measure for customer 
churn prediction. For credit scoring, this implementation assumes an LGD distribution 
with two point masses, and a constant ROI. For churn prediction, this implementation
assumes a beta distribution and a constant CLV.
}
\details{
\tabular{ll}{
Package: \tab EMP\cr
Type: \tab Package\cr
Version: \tab 0.0.1\cr
Date: \tab 2021-05-10\cr
License: \tab GPL (>=3)\cr
}
The package exports only two functions, empCreditScoring and empChurn.
}
\author{
Authors: Cristian Bravo, Seppe vanden Broucke and Thomas Verbraken.
Mantainer: Cristian Bravo <cbravoro@uwo.ca>.
}
\references{
Verbraken, T., Wouter, V. and Baesens, B. (2013). A Novel Profit Maximizing Metric for Measuring Classification Performance of Customer Churn Prediction Models. Knowledge and Data Engineering, IEEE Transactions on. 25 (5): 961-973. Available Online: \url{http://ieeexplore.ieee.org/iel5/69/6486492/06165289.pdf?arnumber=6165289}
Verbraken, T., Bravo, C., Weber, R. and Baesens, B. (2014). Development and application of consumer credit scoring models using profit-based classification measures. European Journal of Operational Research. 238 (2): 505 - 513. Available Online: \url{http://www.sciencedirect.com/science/article/pii/S0377221714003105}
}
\examples{
# Construct artificial probability scores and true class labels
score.ex <- runif(1000, 0, 1)
class.ex <- unlist(lapply(score.ex, function(x){rbinom(1,1,x)}))

# Calculate EMP measures for credit risk scoring
empCreditScoring(score.ex, class.ex)

# Calculate EMP measures for customer churn prediction
empChurn(score.ex, class.ex)
}
