#################################################################################################################################################################################
CODE CURRENTLY UNDERGOING WORK
#################################################################################################################################################################################

def __empRocInfo(probability_scores, true_class_labels):
    
### Initializing variables for rest of function ###

    arguments_list = [probability_scores, true_class_labels]
    factor_array = []
    n_0 = 0
    number_of_factors = 0
    
### Converting both inputs into arrays for further use in function (allows user to input a list, array, matrix or single columns of a data frame) ###
    
    for i in range(0, len(arguments_list)):
        if (type(arguments_list[i]) == pd.DataFrame):
            arguments_list[i] = arguments_list[0].to_numpy().reshape(1,-1)[0]
        if (type(arguments_list[i]) == np.matrix):
            arguments_list[i] = arguments_list[i].reshape(1,-1)[0]
        if (type(arguments_list[i]) == list):
            arguments_list[i] = np.array(arguments_list[i])

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
            n_0 = n_0 + 1
            
### Calculating metrics ###

    total_number_of_observations = len(true_class_labels)
    n_1 = total_number_of_observations - n_0
    pi_0 = n_0 / total_number_of_observations
    pi_1 = n_1 / total_number_of_observations
    fpr, tpr, thresholds = sk.metrics.roc_curve(true_class_labels, probability_scores)
    roc_output = np.c_[fpr, tpr]
    indices_array = sp.spatial.ConvexHull(roc_output).vertices
    def diagonal(x):
        diagonal = 0.5*x
        return diagonal
    F_1 = roc_output[indices_array, 0]
    F_1 = np.sort(F_1)
    diagonal_array = diagonal(F_1)
    F_0 = roc_output[indices_array, 1]
    F_0 = np.sort(F_0)
    for i in range(0, len(F_0)):
        if (F_0[i] < diagonal_array[i]):
            np.delete(F_0, i)
            np.delete(F_1, i)
    
### Returns metrics in an array ###
    
    return [n_0, n_1, pi_0, pi_1, F_0, F_1]
