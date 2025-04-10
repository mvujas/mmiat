import numpy as np

def average_case_accuracy(y_true, y_score):
    """
    Calculate the average case accuracy for a set of true labels and predicted scores.
    The average case accuracy is computed by ordering the samples from lowest to highest confidence,
    then calculating the number of correct matches as the threshold is increased from -inf to +inf.
    This involves summing the number of ones for ground truth 1 and the number of zeros for ground truth 0
    across all possible threshold configurations.
    Args:
        y_true (np.ndarray): Array of true binary labels (0 and 1).
        y_score (np.ndarray): Array of predicted scores.
    Returns:
        float: The average case accuracy.
    """
    # First, we order samples from lowest to highest confidence
    total_samples = y_true.shape[0]
    # print(y_score)
    # print(y_true.shape, y_score.shape)
    y_true_sorted = y_true[np.argsort(y_score)]
    # Now we calculate how many times will the label of predictions match
    # the ground truth as threshold is increase.
    # First we start with all 1s and then switch each leftmost 1 to 0
    # marking a movement of a threshold until there are all 0s.
    # This represents moving the threshold from -inf to +inf.
    # In total there are total_samples + 1 thresholds.
    # Position i (0,1,2,...) will have a 1 in i+1 of them, and 0 in the rest total_thresholds - i configurations.
    # It's a correct match if in both places is 1, for ground truth 1
    #   we sum the number of ones, and for ground truth 0 we sum the number of zeros.
    total_thresholds = total_samples + 1
    one_config_count = np.arange(1, total_thresholds)
    
    total_correct_matches = np.where(
        y_true_sorted,
        one_config_count, # if y_true_sorted is 1, we sum the number of ones
        total_thresholds - one_config_count # if y_true_sorted is 0, we sum the number of zeros
    ).sum()
    # For each of the total_thresholds thresholds, we have total_samples possible spaces for matches
    total_possible_matches = total_samples * total_thresholds
    average_accuracy = total_correct_matches / total_possible_matches
    return average_accuracy