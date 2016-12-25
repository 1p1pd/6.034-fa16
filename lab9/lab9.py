# MIT 6.034 Lab 9: Boosting (Adaboost)
# Written by Jessica Noss (jmn), Dylan Holmes (dxh), and 6.034 staff

from math import log as ln
from utils import *


#### BOOSTING (ADABOOST) #######################################################

def initialize_weights(training_points):
    """Assigns every training point a weight equal to 1/N, where N is the number
    of training points.  Returns a dictionary mapping points to weights."""
    result = {}
    for p in training_points:
        result[p] = make_fraction(1, len(training_points))
    return result

def calculate_error_rates(point_to_weight, classifier_to_misclassified):
    """Given a dictionary mapping training points to their weights, and another
    dictionary mapping classifiers to the training points they misclassify,
    returns a dictionary mapping classifiers to their error rates."""
    result = {}
    for c in classifier_to_misclassified:
        misspoint = classifier_to_misclassified[c]
        result[c] = 0
        for p in misspoint:
            result[c] += point_to_weight[p]
    return result

def pick_best_classifier(classifier_to_error_rate, use_smallest_error=True):
    """Given a dictionary mapping classifiers to their error rates, returns the
    best* classifier, or raises NoGoodClassifiersError if best* classifier has
    error rate 1/2.  best* means 'smallest error rate' if use_smallest_error
    is True, otherwise 'error rate furthest from 1/2'."""
    if use_smallest_error:
        bc = sorted(classifier_to_error_rate,
                    key=lambda x: (classifier_to_error_rate[x], x))[0]
    else:
        bc = sorted(classifier_to_error_rate,
                    key=lambda x: (-abs(classifier_to_error_rate[x] - Fraction(1, 2)), x))[0]
    if classifier_to_error_rate[bc] == Fraction(1, 2):
        raise NoGoodClassifiersError("No good classifier!")
    else:
        return bc

def calculate_voting_power(error_rate):
    """Given a classifier's error rate (a number), returns the voting power
    (aka alpha, or coefficient) for that classifier."""
    if error_rate == 0:
        return INF
    elif error_rate == 1:
        return -INF
    else:
        return 0.5 * ln((1 - error_rate) / error_rate)

def get_overall_misclassifications(H, training_points, classifier_to_misclassified):
    """Given an overall classifier H, a list of all training points, and a
    dictionary mapping classifiers to the training points they misclassify,
    returns a set containing the training points that H misclassifies.
    H is represented as a list of (classifier, voting_power) tuples."""
    result = set()
    for p in training_points:
        v = 0
        for c in H:
            if p in classifier_to_misclassified[c[0]]:
                v -= c[1]
            else:
                v += c[1]
        if v <= 0:
            result.add(p)
    return result

def is_good_enough(H, training_points, classifier_to_misclassified, mistake_tolerance=0):
    """Given an overall classifier H, a list of all training points, a
    dictionary mapping classifiers to the training points they misclassify, and
    a mistake tolerance (the maximum number of allowed misclassifications),
    returns False if H misclassifies more points than the tolerance allows,
    otherwise True.  H is represented as a list of (classifier, voting_power)
    tuples."""
    return len(get_overall_misclassifications(H, training_points, classifier_to_misclassified)) <= mistake_tolerance

def update_weights(point_to_weight, misclassified_points, error_rate):
    """Given a dictionary mapping training points to their old weights, a list
    of training points misclassified by the current weak classifier, and the
    error rate of the current weak classifier, returns a dictionary mapping
    training points to their new weights.  This function is allowed (but not
    required) to modify the input dictionary point_to_weight."""
    result = {}
    for p in point_to_weight:
        if p in misclassified_points:
            result[p] = make_fraction(1, 2) * point_to_weight[p] / error_rate
        else:
            result[p] = make_fraction(1, 2) * point_to_weight[p] / (1 - error_rate)
    return result

def adaboost(training_points, classifier_to_misclassified,
             use_smallest_error=True, mistake_tolerance=0, max_rounds=INF):
    """Performs the Adaboost algorithm for up to max_rounds rounds.
    Returns the resulting overall classifier H, represented as a list of
    (classifier, voting_power) tuples."""
    count = 0
    weights = initialize_weights(training_points)
    H = []
    while count < max_rounds:
        error = calculate_error_rates(weights, classifier_to_misclassified)
        best = None
        try:
            best = pick_best_classifier(error, use_smallest_error)
        except NoGoodClassifiersError:
            return H
        e = 0
        for p in classifier_to_misclassified[best]:
            e += weights[p]
        vp = calculate_voting_power(e)
        H.append((best, vp))
        if is_good_enough(H, training_points, classifier_to_misclassified, mistake_tolerance):
            return H
        weights = update_weights(weights, classifier_to_misclassified[best], e)
        count += 1
    return H

#### SURVEY ####################################################################

NAME = None
COLLABORATORS = None
HOW_MANY_HOURS_THIS_LAB_TOOK = None
WHAT_I_FOUND_INTERESTING = None
WHAT_I_FOUND_BORING = None
SUGGESTIONS = None
