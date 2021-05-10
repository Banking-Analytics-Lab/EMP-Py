def empChurn(probability_scores, true_class_labels, alpha=6, beta=14, clv=200, d=10, f=1, print_output=True, return_output=True, rounding=None):
  
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
  #   print_output: Boolean variable that determines if output is printed. If True, output will be printed
  #   return_output: Boolean variable that determines if output is returned. If True, output will be returned
  #   rounding: An integer value that determines the precision of the output. The integer determines the decimal places retained in the output
  #
  # Value:
  #   An EMP object with four components.
  #     MP: The Maximum Profit of the ROC curve at MP_fraction cutoff
  #     MP_fraction: The percentage of cases that should be excluded, that is, 
  #     the percentual cutoff at MP profit.
  #     EMP: The Expected Maximum Profit of the ROC curve at EMP_fraction cutoff.}
  #     EMP_fraction: The percentage of cases that should be excluded, that is, 
  #     the percentual cutoff at EMP profit.
    
    def B(a, b, X):
        return betainc(a, b, X)* BETA(a, b)

### Initializing constants and variables for rest of function ###

    roc = __empRocInfo(probability_scores, true_class_labels)
    E_GAMMA = alpha / (alpha + beta)
    DELTA = d / clv
    PHI = f / clv
    
### Calculating MP and MP fraction ###

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
