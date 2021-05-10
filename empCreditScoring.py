def empCreditScoring(probability_scores, true_class_labels, p_0=0.55, p_1=0.1, ROI=0.2644, print_output=True, return_output=True, rounding=None):
    
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
  #   print_output: Boolean variable that determines if output is printed. If True, output will be printed
  #   return_output: Boolean variable that determines if output is returned. If True, output will be returned
  #   rounding: An integer value that determines the precision of the output. The integer determines the decimal places retained in the output
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
