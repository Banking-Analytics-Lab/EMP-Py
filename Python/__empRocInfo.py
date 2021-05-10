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
            n_0 = n_0 + 1
            
### Calculating metrics ###

    total_number_of_observations = len(arguments_list[1])
    n_1 = total_number_of_observations - n_0
    pi_1 = n_1 / total_number_of_observations
    pi_0 = n_0 / total_number_of_observations
    fpr, tpr, thresholds = sk.metrics.roc_curve(arguments_list[1], arguments_list[0])
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
