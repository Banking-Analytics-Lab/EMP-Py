#################################################################################################################################################################################
CODE CURRENTLY UNDERGOING WORK
#################################################################################################################################################################################

def empCreditScoring(probability_scores, true_class_labels, p_0 = 0.55, p_1 = 0.1, ROI = 0.2644):
    roc = __empRocInfo(probability_scores, true_class_labels)
    alpha = 1 - p_0 - p_1
    (roc[3]*ROI/roc[2])*np.diff(roc[5])/np.diff(roc[4])
    lambda_values = np.append(0, (roc[3]*ROI/roc[2])*np.diff(roc[5])/np.diff(roc[4]))
    lambda_values = np.append(lambda_values[lambda_values < 1], 1)
    lambdaii = lambda_values[range(0, len(lambda_values)-1)]
    lambdaie = lambda_values[range(1, len(lambda_values))]
    F_0 = (roc[4])[range(0, len(lambdaii))]
    F_1 = (roc[5])[range(0, len(lambdaii))]
    EMPC = np.sum(alpha*(lambdaie - lambdaii)*(roc[2]*F_0*(lambdaie + lambdaii)/2 - ROI*F_1*roc[3])) + (roc[2]*F_0[len(F_0)-1] - ROI*roc[3]*F_1[len(F_1)-1])*p_1
    EMPC_fraction = np.sum(alpha*(lambdaie - lambdaii)*(roc[2]*F_0 + roc[3]*F_1)) + p_1*(roc[2]*F_0[len(F_0)-1] + roc[3]*F_1[len(F_1)-1])
    
    return [EMPC, EMPC_fraction]
