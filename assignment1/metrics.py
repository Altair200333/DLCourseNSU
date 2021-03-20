import numpy as np
import collections

def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''
    
    TP = np.sum([a and b for a,b in zip(prediction, ground_truth)])
    TN = np.sum([not a and not b for a,b in zip(prediction, ground_truth)])
    FP = np.sum([a and not b for a,b in zip(prediction, ground_truth)])
    FN = np.sum([not a and b for a,b in zip(prediction, ground_truth)])

    accuracy = (TP+TN)/len(ground_truth)
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    f1 = 2*precision*recall/(precision+recall)
    
    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score
    
    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    # TODO: Implement computing accuracy
    
    return np.where(prediction == ground_truth)[0].shape[0]/prediction.shape[0]
