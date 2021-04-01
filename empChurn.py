#################################################################################################################################################################################
CODE CURRENTLY UNDERGOING WORK
#################################################################################################################################################################################

def empChurn(probability_scores, true_class_labels, alpha = 6, beta = 14, clv = 200, d = 10, f = 1):
    roc = __empRocInfo(probability_scores, true_class_labels)
    e_gamma = alpha/(alpha + beta)
    delta = d/clv
    phi = f/clv
    (roc[3]*(delta + phi)*np.diff(roc[5]) + roc[2]*phi*np.diff(roc[4])) / (roc[2]*(1 - delta)*np.diff(roc[4]))
    gamma_values = np.append(0, (roc[3]*(delta + phi)*np.diff(roc[5]) + roc[2]*phi*np.diff(roc[4])) / (roc[2]*(1 - delta)*np.diff(roc[4])))
    gamma_values = np.append(gamma_values[gamma_values < 1], 1)
    ind_E = np.max(np.where((gamma_values < e_gamma) == True))
    MP = clv*((e_gamma*(1 - delta) - phi)*roc[2]*(roc[4])[ind_E] - (delta + phi)*roc[3]*(roc[5])[ind_E])
    MP_fraction = roc[2]*(roc[4])[ind_E] + roc[3]*(roc[5])[ind_E]
    gammaii = gamma_values[range(0, len(gamma_values)-1)]
    gammaie = gamma_values[range(1, len(gamma_values))]
    F_0 = (roc[4])[range(0, len(gammaii))]
    F_1 = (roc[5])[range(0, len(gammaii))]
    def B(a, b, x):
        return (sp.stats.beta.cdf(x, a, b)*sp.special.beta(a, b))
    contr_0 = (clv*(1 - delta)*roc[2]*F_0)*(B(alpha + 1, beta, gammaie) - B(alpha + 1, beta, gammaii)) / B(alpha, beta, 1)
    contr_1 = (-clv*(phi*roc[2]*F_0 + (delta + phi)*roc[3]*F_1))*(B(alpha, beta, gammaie) - B(alpha, beta, gammaii)) / B(alpha, beta, 1)
    EMP = np.sum(contr_0 + contr_1)
    EMP_fraction = np.matmul((((B(alpha, beta, gammaie) - B(alpha, beta, gammaii)) / B(alpha, beta, 1)).T), (roc[2]*F_0 + roc[3]*F_1))
    
    return [MP, MP_fraction, EMP, EMP_fraction]
