\name{empCreditScoring}
\alias{empCreditScoring}
\title{empCreditScoring}
\description{
Estimates the EMP for credit risk scoring, considering constant ROI and a bimodal LGD function with point masses p0 and p1 for no loss and total loss, respectively.
}
\usage{
empCreditScoring(probability_scores, true_class_labels, p_0=0.55, p_1=0.1, ROI=0.2644, print_output=True, return_output=True, rounding=None):
}
\arguments{
\item{probability_scores}{A list, array, matrix or single pandas dataframe column of probability scores.}
\item{true_class_labels}{A list, array, matrix or single pandas dataframe column of true class labels.}
\item{p_0}{Percentage of cases on the first point mass of the LGD distribution (complete recovery).}
\item{p_1}{Percentage of cases on the second point mass of the LGD distribution (complete loss).}
\item{ROI}{Constant ROI per granted loan. A percentage.}
\item{print_output}{Boolean variable that determines if output is printed. If True, output will be printed.}
\item{return_output}{Boolean variable that determines if output is returned. If True, output will be returned.}
\item{rounding}{An integer value that determines the precision of the output. The integer determines the decimal places retained in the output.}
}
\value{
An EMP object with two components.
\item{EMPC}{The Expected Maximum Profit of the ROC curve at EMP_fraction cutoff.}
\item{EMPC_fraction}{The percentage of cases that should be excluded, that is, the percentual cutoff at EMP profit.}
}
\references{
Verbraken, T., Wouter, V. and Baesens, B. (2013). A Novel Profit Maximizing Metric for Measuring Classification Performance of Customer Churn Prediction Models. Knowledge and Data Engineering, IEEE Transactions on. 25 (5): 961-973. Available Online: \url{http://ieeexplore.ieee.org/iel5/69/6486492/06165289.pdf?arnumber=6165289}
Verbraken, T., Bravo, C., Weber, R. and Baesens, B. (2014). Development and application of consumer credit scoring models using profit-based classification measures. European Journal of Operational Research. 238 (2): 505 - 513. Available Online: \url{http://www.sciencedirect.com/science/article/pii/S0377221714003105}
}
\author{
Cristian Bravo, Seppe vanden Broucke and Thomas Verbraken.
}
\seealso{
See Also \code{\link{empChurn}}, \code{\link{prediction}}.
}
\examples{
# Construct artificial probability scores and true class labels
score.ex <- runif(1000, 0, 1)
class.ex <- unlist(lapply(score.ex, function(x){rbinom(1,1,x)}))

# Calculate EMP measures for credit risk scoring
empCreditScoring(score.ex, class.ex)

# Calculate EMP measures for credit risk scoring with point masses
# in 0.1 and 0.9, and 0.1 ROI
empCreditScoring(score.ex, class.ex, 0.1, 0.1, 0.1)
}
\keyword{classif}
\keyword{evaluation}
