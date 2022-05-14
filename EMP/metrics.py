import numpy as np 
import pandas as pd
from sklearn.metrics import roc_curve
import collections
from scipy.spatial import ConvexHull
from scipy.special import betainc
import scipy.special as ss
import sys

def __empRocInfo(probability_scores, true_class_labels):

    # This software comes with absolutely no warranty. Use at your own risk.
    #
    # Provides information related to the ROC given probability and class label vectors. 
    # This function is not to be called directly in a normal use case. 
    # Instead, the other functions in this package call this function when necessary.
    #
    #
    # Arguments:
    #   probability_scores: A list, array, matrix or single pandas dataframe column of probability scores.
    #   true_class_labels: A list, array, matrix or single pandas dataframe column of true class labels.
    #
    # Value:
    #   A RocInfo object with six components:
    #     n_0: Number of positive observations.
    #     n_1: Number of negative observations.
    #     pi_0: Prior probability of positive observation.
    #     pi_1: Prior probability of negative observation.
    #     F_0: Convex hull of ROC y values.
    #     F_1: Convex hull of ROC x values.
        
    ### Initializing variables for rest of function ###

    arguments_list = [probability_scores, true_class_labels]
    factor_array = []
    n_0 = 0
    number_of_factors = 0
    
    def linear_line(m, x, b):
        # Calculates y components of a linear line given slope, intercept and x component
        #
        # Arguments:
        #   m: Constant slope value of linear line
        #   x: Constant x component values of linear line
        #   b: Constant intercept value of linear line
        # Value:
        # m*x + b: Constant y component of linear line
        return m*x + b
    
### Converting both inputs into arrays for further use in function (allows user to input a list, array, matrix or single columns of a data frame) ###
    
    for i in range(0, len(arguments_list)):
        if (type(arguments_list[i]) == pd.Series):
            arguments_list[i] = arguments_list[i].to_numpy().reshape(1,-1)[0]
        if (type(arguments_list[i]) == np.matrix):
            arguments_list[i] = np.asarray(arguments_list[i]).reshape(1,-1)[0]
        if (type(arguments_list[i]) == list):
            arguments_list[i] = np.array(arguments_list[i]).reshape(1,-1)[0]

### Checking that the data is appropriate for binary classification ###

    for i in range(0, len(arguments_list[1])):
        if (arguments_list[1][i] not in factor_array):
            factor_array.append(arguments_list[1][i])
            number_of_factors = number_of_factors + 1
            
    if (number_of_factors > 2):
        sys.exit("More than 2 classes in true class labels, data not suitable for binary classification")

### Checking to ensure that the probability scores array is the same length as the true class labels array ###

    if (len(arguments_list[0]) != len(arguments_list[1])):
        sys.exit("Length of input arrays probability_scores and true_class_labels are not equal")

### Checking that there are no invalid probabilities in the probability scores array ###

    for i in range(0, len(arguments_list[0])):
        if (arguments_list[0][i] > 1 or arguments_list[0][i] < 0):
            sys.exit("Invalid Probability Score: You have a probability outside [0,1]")

### Counting the number of positive cases in the true class values array ###

    for i in range(0, len(arguments_list[1])):
        if (arguments_list[1][i] == 1):
            n_0 +=  1
            
### Calculating metrics ###

    total_number_of_observations = len(arguments_list[1])
    n_1 = total_number_of_observations - n_0
    pi_1 = n_1 / total_number_of_observations
    pi_0 = n_0 / total_number_of_observations
    fpr, tpr, thresholds = roc_curve(arguments_list[1], arguments_list[0])
    roc_output = np.c_[fpr, tpr]
    hull = ConvexHull(roc_output)
    vertices = []
    
    for k in hull.vertices:
        if roc_output[k,1] >= linear_line(1, roc_output[k,0], 0):
            vertices.append(k)
            
    F_1 = roc_output[vertices, 0]
    F_1 = np.sort(F_1)
    F_0 = roc_output[vertices, 1]
    F_0 = np.sort(F_0)
    
    return n_0, n_1, pi_0, pi_1, F_0, F_1








def empCreditScoring(probability_scores, true_class_labels, p_0=0.55, p_1=0.1, ROI=0.2644, print_output=False, return_output=True, rounding=None):

    # This software comes with absolutely no warranty. Use at your own risk.
    #
    # Adapted from:
    # Verbraken T., Bravo C., Weber, R., and Baesens, B. 2014. Development and 
    # application of consumer credit scoring models using profit-based 
    # classification measures. European Journal of Operational Research.
    # 238 (2): 505-513.
    #
    # Estimates the EMP for credit risk scoring, considering constant ROI and
    # a bimodal LGD function with point masses p0 and p1 for no loss and total 
    # loss, respectively.
    #
    #
    # Arguments:
    #   probability_scores: A list, array, matrix or single pandas dataframe column of probability scores.
    #   true_class_labels: A list, array, matrix or single pandas dataframe column of true class labels.
    #   p_0: Percentage of cases on the first point mass of the LGD distribution (complete recovery).
    #   p_1: Percentage of cases on the second point mass of the LGD distribution (complete loss).
    #   ROI: Constant ROI per granted loan. A percentage.
    #   print_output: Boolean variable that determines if output is printed. If True, output will be printed.
    #   return_output: Boolean variable that determines if output is returned. If True, output will be returned.
    #   rounding: An integer value that determines the precision of the output. The integer determines the decimal places retained in the output.
    #
    # Value:
    #   An EMP object with two components.
    #     EMPC: The Expected Maximum Profit of the ROC curve at EMP_fraction cutoff.
    #     EMPC_fraction: The percentage of cases that should be excluded, that is, 
    #     the percentual cutoff at EMP profit.
        
    ### Initializing constants and variables for rest of function ###

        roc = __empRocInfo(probability_scores, true_class_labels)
        ALPHA = 1 - p_0 - p_1
        
    ### Calculating metrics ###

        lambda_values = np.append(0, (((roc[3]*ROI) / roc[2])*(np.diff(roc[5]) / np.diff(roc[4]))))
        lambda_values = np.append(lambda_values[lambda_values < 1], 1)
        lambdaii = lambda_values[range(0, len(lambda_values) - 1)]
        lambdaie = lambda_values[range(1, len(lambda_values))]
        F_0 = (roc[4])[range(0, len(lambdaii))]
        F_1 = (roc[5])[range(0, len(lambdaii))]
        
        EMPC = np.sum(ALPHA*(lambdaie - lambdaii)*(roc[2]*F_0*(lambdaie + lambdaii) / 2 - ROI*F_1*roc[3])) + (roc[2]*F_0[len(F_0) - 1] - ROI*roc[3]*F_1[len(F_1) - 1])*p_1
        EMPC_fraction = np.sum(ALPHA*(lambdaie - lambdaii)*(roc[2]*F_0 + roc[3]*F_1)) + p_1*(roc[2]*F_0[len(F_0) - 1] + roc[3]*F_1[len(F_1) - 1])

    ### Formatting output ###

        if rounding != None:
            EMPC = np.round(EMPC, decimals = rounding)
            EMPC_fraction = np.round(EMPC_fraction, decimals = rounding)
            
        output = collections.namedtuple('output',['EMPC','EMPC_fraction'])
        output = output(EMPC, EMPC_fraction)
        
        if print_output:
            print("******************************************************************************")
            print("EMP Credit Scoring Output")
            print("******************************************************************************")
            print("EMPC: ", EMPC)
            print("")
            print("EMPC Fraction: ", EMPC_fraction)
            
        if return_output:
            return output
        
            
            
            
            
    
    
    
def empChurn(probability_scores, true_class_labels, alpha=6, beta=14, clv=200, d=10, f=1, print_output=False, return_output=True, rounding=None):

    # This software comes with absolutely no warranty. Use at your own risk.
    #
    # Adapted from:
    # Verbraken, T., Wouter, V. and Baesens, B. (2013). A Novel Profit Maximizing 
    # Metric for Measuring Classification Performance of Customer Churn Prediction
    # Models. Knowledge and Data Engineering, IEEE Transactions on. 25 (5): 
    # 961-973.
    # Available Online: http://ieeexplore.ieee.org/iel5/69/6486492/06165289.pdf
    #
    # Estimates the EMP for customer churn prediction, considering constant CLV
    # and a given cost of contact f and retention offer d.
    #
    #
    # Arguments:
    #   probability_scores: A list, array, matrix or single pandas dataframe column of probability scores.
    #   true_class_labels: A list, array, matrix or single pandas dataframe column of true class labels.
    #   alpha: Alpha parameter of unimodel beta distribution.
    #   beta: Beta parameter of unimodel beta distribution.
    #   clv: Constant CLV per retained customer.
    #   d: Constant value of retention offer.
    #   f: Constant cost of contact.
    #   print_output: Boolean variable that determines if output is printed. If True, output will be printed.
    #   return_output: Boolean variable that determines if output is returned. If True, output will be returned.
    #   rounding: An integer value that determines the precision of the output. The integer determines the decimal places retained in the output.
    #
    # Value:
    #   An EMP object with four components.
    #     MP: The Maximum Profit of the ROC curve at MP_fraction cutoff.
    #     MP_fraction: The percentage of cases that should be excluded, that is, 
    #     the percentual cutoff at MP profit.
    #     EMP: The Expected Maximum Profit of the ROC curve at EMP_fraction cutoff.
    #     EMP_fraction: The percentage of cases that should be excluded, that is, 
    #     the percentual cutoff at EMP profit.
        
        def B(a, b, X):
            return betainc(a, b, X)* ss.beta(a, b)

    ### Initializing constants and variables for rest of function ###

        roc = __empRocInfo(probability_scores, true_class_labels)
        E_GAMMA = alpha / (alpha + beta)
        DELTA = d / clv
        PHI = f / clv
        
    ### Calculating MP and MP fraction ###
        '''
        n_0, n_1, pi_0, pi_1, F_0, F_1
        gamma values = pi_0 (delta * phi)
        '''
        gamma_values = np.append(0, (roc[3]*(DELTA + PHI)*np.diff(roc[5]) + roc[2]*PHI*np.diff(roc[4])) / (roc[2]*(1 - DELTA)*np.diff(roc[4])))
        gamma_values = np.append(gamma_values[gamma_values < 1], 1)
        ind_E = np.max(np.where((gamma_values < E_GAMMA) == True))
        MP = clv*((E_GAMMA*(1 - DELTA) - PHI)*roc[2]*(roc[4])[ind_E] - (DELTA + PHI)*roc[3]*(roc[5])[ind_E])
        MP_fraction = roc[2]*(roc[4])[ind_E] + roc[3]*(roc[5])[ind_E]

    ### Calculating EMP and EMP fraction ###

        gammaii = gamma_values[range(0, len(gamma_values) - 1)]
        gammaie = gamma_values[range(1, len(gamma_values))]
        F_0 = (roc[4])[range(0, len(gammaii))]
        F_1 = (roc[5])[range(0, len(gammaii))]
        contr_0 = (clv*(1 - DELTA)*roc[2]*F_0)*(B(alpha + 1, beta, gammaie) - B(alpha + 1, beta, gammaii)) / B(alpha, beta, 1)
        contr_1 = (-clv*(PHI*roc[2]*F_0 + (DELTA + PHI)*roc[3]*F_1))*(B(alpha, beta, gammaie) - B(alpha, beta, gammaii)) / B(alpha, beta, 1)
        EMP = np.sum(contr_0 + contr_1)
        EMP_fraction = np.matmul((((B(alpha, beta, gammaie) - B(alpha, beta, gammaii)) / B(alpha, beta, 1)).T), (roc[2]*F_0 + roc[3]*F_1))

    ### Formatting output ###

        if rounding != None:
            MP = np.round(MP, decimals = rounding)
            MP_fraction = np.round(MP_fraction, decimals = rounding)
            EMP = np.round(EMP, decimals = rounding)
            EMP_fraction = np.round(EMP_fraction, decimals = rounding)
            
        output = collections.namedtuple('output',['MP','MP_fraction','EMP', 'EMP_fraction'])
        output = output(MP, MP_fraction, EMP, EMP_fraction)
        
        if print_output:
            print("******************************************************************************")
            print("EMP Churn Output")
            print("******************************************************************************")
            print("MP: ", MP)
            print("")
            print("MP Fraction: ", MP_fraction)
            print("")
            print("EMP: ", EMP)
            print("")
            print("EMP Fraction: ", EMP_fraction)
            
        if return_output:
            return output
