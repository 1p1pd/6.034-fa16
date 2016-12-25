# MIT 6.034 Lab 7: Support Vector Machines
# Written by Jessica Noss (jmn) and 6.034 staff

from svm_data import *

# Vector math
def dot_product(u, v):
    """Computes dot product of two vectors u and v, each represented as a tuple
    or list of coordinates.  Assume the two vectors are the same length."""
    return sum(i[0] * i[1] for i in zip(u, v))

def norm(v):
    "Computes length of a vector v, represented as a tuple or list of coords."
    return dot_product(v, v) ** 0.5

# Equation 1
def positiveness(svm, point):
    "Computes the expression (w dot x + b) for the given point"
    return dot_product(svm.w, point.coords) + svm.b

def classify(svm, point):
    """Uses given SVM to classify a Point.  Assumes that point's true
    classification is unknown.  Returns +1 or -1, or 0 if point is on boundary"""
    result = positiveness(svm, point)
    if result > 0:
        return 1
    elif result < 0:
        return -1
    else:
        return 0

# Equation 2
def margin_width(svm):
    "Calculate margin width based on current boundary."
    return 2 / norm(svm.w)

# Equation 3
def check_gutter_constraint(svm):
    """Returns the set of training points that violate one or both conditions:
        * gutter constraint (positiveness == classification for support vectors)
        * training points must not be between the gutters
    Assumes that the SVM has support vectors assigned."""
    result = set()
    for point in svm.training_points:
        if point in svm.support_vectors and positiveness(svm, point) != point.classification:
            result.add(point)
        elif point not in svm.support_vectors and abs(positiveness(svm, point)) < 1:
            result.add(point)
    return result

# Equations 4, 5
def check_alpha_signs(svm):
    """Returns the set of training points that violate either condition:
        * all non-support-vector training points have alpha = 0
        * all support vectors have alpha > 0
    Assumes that the SVM has support vectors assigned, and that all training
    points have alpha values assigned."""
    result = set()
    for point in svm.training_points:
        if point in svm.support_vectors and point.alpha <= 0:
            result.add(point)
        elif point not in svm.support_vectors and point.alpha != 0:
            result.add(point)
    return result

def check_alpha_equations(svm):
    """Returns True if both Lagrange-multiplier equations are satisfied,
    otherwise False.  Assumes that the SVM has support vectors assigned, and
    that all training points have alpha values assigned."""
    lar = None
    larnum = None
    for point in svm.support_vectors:
        try:
            lar += point.alpha * point.classification
        except TypeError:
            lar = point.alpha * point.classification
        try:
            larnum = vector_add(larnum, scalar_mult(point.alpha * point.classification, point.coords))
        except TypeError:
            larnum = scalar_mult(point.alpha * point.classification, point.coords)
    if lar != 0:
        return False
    for i in range(len(larnum)):
        if larnum[i] != svm.w[i]:
            return False
    return True

# Classification accuracy
def misclassified_training_points(svm):
    """Returns the set of training points that are classified incorrectly
    using the current decision boundary."""
    result = set()
    for point in svm.training_points:
        if classify(svm, point) != point.classification:
            result.add(point)
    return result

# Training
def update_svm_from_alphas(svm):
    """Given an SVM with training data and alpha values, use alpha values to
    update the SVM's support vectors, w, and b.  Return the updated SVM."""
    w = None
    for point in svm.training_points:
        if point.alpha > 0:
            try:
                w = vector_add(w, scalar_mult(point.alpha * point.classification, point.coords))
            except TypeError:
                w = scalar_mult(point.alpha * point.classification, point.coords)
            if point not in svm.support_vectors:
                svm.support_vectors.append(point)
    for point in svm.support_vectors:
        if point.alpha == 0:
            svm.support_vectors.remove(point)
    svm.w = w
    negmin = None
    posmax = None
    for point in svm.training_points:
        if point.alpha > 0:
            if point.classification == -1:
                if negmin is None:
                    negmin = -dot_product(w, point.coords)
                elif negmin > -dot_product(w, point.coords):
                    negmin = -dot_product(w, point.coords)
            elif point.classification == 1:
                if posmax is None:
                    posmax = -dot_product(w, point.coords)
                elif posmax < -dot_product(w, point.coords):
                    posmax = -dot_product(w, point.coords)
    svm.b = (negmin + posmax) / 2
    return svm

# Multiple choice
ANSWER_1 = 11
ANSWER_2 = 6
ANSWER_3 = 3
ANSWER_4 = 2

ANSWER_5 = 'AD'
ANSWER_6 = 'ABD'
ANSWER_7 = 'ABD'
ANSWER_8 = []
ANSWER_9 = 'ABD'
ANSWER_10 = 'ABD'

ANSWER_11 = False
ANSWER_12 = True
ANSWER_13 = False
ANSWER_14 = False
ANSWER_15 = False
ANSWER_16 = True

ANSWER_17 = [1, 3, 8]
ANSWER_18 = [1, 2, 4, 5, 6, 7, 8]
ANSWER_19 = [1, 2, 4, 5, 6, 7, 8]

ANSWER_20 = 6


#### SURVEY ####################################################################

NAME = "Yifan Wang"
COLLABORATORS = ""
HOW_MANY_HOURS_THIS_LAB_TOOK = 1
WHAT_I_FOUND_INTERESTING = "Everything"
WHAT_I_FOUND_BORING = ""
SUGGESTIONS = None
