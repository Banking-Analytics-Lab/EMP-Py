\name{empChurn}
\alias{empChurn}
\title{empChurn}
\description{
Estimates the EMP for customer churn prediction, considering constant CLV and a given cost of contact f and retention offer d.
}
\usage{
empChurn(probability_scores, true_class_labels, alpha=6, 
beta=14, clv=200, d=10, f=1, print_output=True, return_output=True, rounding=None)
}
\arguments{
\item{probability_scores}{A list, array, matrix or single pandas dataframe column of probability scores.}
\item{true_class_labels}{A list, array, matrix or single pandas dataframe column of true class labels.}
\item{alpha}{Alpha parameter of unimodel beta distribution.}
\item{beta}{Beta parameter of unimodel beta distribution.}
\item{clv}{Constant CLV per retained customer.}
\item{d}{Constant value of retention offer.}
\item{f}{Constant cost of contact.}
\item{print_output}{Boolean variable that determines if output is printed. If True, output will be printed.}
\item{return_output}{Boolean variable that determines if output is returned. If True, output will be returned.}
\item{rounding}{An integer value that determines the precision of the output. The integer determines the decimal places retained in the output.}
}
\value{
An EMP object with four components.
\item{MP}{The Maximum Profit of the ROC curve at MP_fraction cutoff.}
\item{MP_fraction}{The percentage of cases that should be excluded, that is, the percentual cutoff at MP profit.}
\item{EMP}{The Expected Maximum Profit of the ROC curve at EMP_fraction cutoff.}
\item{EMP_fraction}{The percentage of cases that should be excluded, that is, the percentual cutoff at EMP profit.}
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

# Calculate EMP measures for customer churn prediction
empChurn(score.ex, class.ex)

# Calculate EMP measures for customer churn prediction with
# lower clv and higher costs
empChurn(score.ex, class.ex, clv = 100, d = 30, f = 5)
}
\keyword{classif}
\keyword{evaluation}
