# MIT 6.034 Lab 5: k-Nearest Neighbors and Identification Trees
# Written by Jessica Noss (jmn), Dylan Holmes (dxh), and Jake Barnwell (jb16)

from api import *
from data import *
import math
log2 = lambda x: math.log(x, 2)
INF = float('inf')

################################################################################
############################# IDENTIFICATION TREES #############################
################################################################################

def id_tree_classify_point(point, id_tree):
    """Uses the input ID tree (an IdentificationTreeNode) to classify the point.
    Returns the point's classification."""
    while not id_tree.is_leaf():
        id_tree = id_tree.apply_classifier(point)
    return id_tree.get_node_classification()

def split_on_classifier(data, classifier):
    """Given a set of data (as a list of points) and a Classifier object, uses
    the classifier to partition the data.  Returns a dict mapping each feature
    values to a list of points that have that value."""
    result = {}
    for i in data:
        try:
            result[classifier.classify(i)].append(i)
        except KeyError:
            result[classifier.classify(i)] = [i]
    return result

#### CALCULATING DISORDER

def branch_disorder(data, target_classifier):
    """Given a list of points representing a single branch and a Classifier
    for determining the true classification of each point, computes and returns
    the disorder of the branch."""
    result = split_on_classifier(data, target_classifier)
    total = 0
    sum = 0
    for item in result:
        total += len(result[item])
    for item in result:
        temp = float(len(result[item])) / float(total)
        sum -= temp * log2(temp)
    return sum

def average_test_disorder(data, test_classifier, target_classifier):
    """Given a list of points, a feature-test Classifier, and a Classifier
    for determining the true classification of each point, computes and returns
    the disorder of the feature-test stump."""
    dic = {}
    count = {}
    total = 0
    for i in data:
        try:
            dic[test_classifier.classify(i)].append(i)
            count[test_classifier.classify(i)] += 1
        except KeyError:
            dic[test_classifier.classify(i)] = [i]
            count[test_classifier.classify(i)] = 1
        total += 1
    result = 0
    for i in dic:
        result += (float(count[test_classifier.classify(dic[i][0])]) / float(total)) * branch_disorder(dic[i], target_classifier)
    return result

## To use your functions to solve part A2 of the "Identification of Trees"
## problem from 2014 Q2, uncomment the lines below and run lab5.py:
#for classifier in tree_classifiers:
#    print classifier.name, average_test_disorder(tree_data, classifier, feature_test("tree_type"))


#### CONSTRUCTING AN ID TREE

def find_best_classifier(data, possible_classifiers, target_classifier):
    """Given a list of points, a list of possible Classifiers to use as tests,
    and a Classifier for determining the true classification of each point,
    finds and returns the classifier with the lowest disorder.  Breaks ties by
    preferring classifiers that appear earlier in the list.  If the best
    classifier has only one branch, raises NoGoodClassifiersError."""
    min = INF
    result = None
    for classifier in possible_classifiers:
        if average_test_disorder(data, classifier, target_classifier) < min:
            result = classifier
            min = average_test_disorder(data, classifier, target_classifier)
    if min == 1.0:
        raise NoGoodClassifiersError
    return result

## To find the best classifier from 2014 Q2, Part A, uncomment:
#print find_best_classifier(tree_data, tree_classifiers, feature_test("tree_type"))


def construct_greedy_id_tree(data, possible_classifiers, target_classifier, id_tree_node=None):
    """Given a list of points, a list of possible Classifiers to use as tests,
    a Classifier for determining the true classification of each point, and
    optionally a partially completed ID tree, returns a completed ID tree by
    adding classifiers and classifications until either perfect classification
    has been achieved, or there are no good classifiers left."""
    if id_tree_node is None:
        id_tree_node = IdentificationTreeNode(target_classifier)
        construct_greedy_id_tree(data, possible_classifiers, target_classifier, id_tree_node)
    else:
        if branch_disorder(data, target_classifier) == 0.0:
            id_tree_node.set_node_classification(target_classifier.classify(data[0]))
        else:
            try:
                bestclassifier = find_best_classifier(data, possible_classifiers, target_classifier)
            except NoGoodClassifiersError:
                return
            id_tree_node = id_tree_node.set_classifier_and_expand(bestclassifier, split_on_classifier(data, bestclassifier))
            for i in id_tree_node.get_branches():
                construct_greedy_id_tree(split_on_classifier(data, bestclassifier)[i], possible_classifiers, target_classifier, id_tree_node.get_branches()[i])
    return id_tree_node

## To construct an ID tree for 2014 Q2, Part A:
#print construct_greedy_id_tree(tree_data, tree_classifiers, feature_test("tree_type"))

## To use your ID tree to identify a mystery tree (2014 Q2, Part A4):
#tree_tree = construct_greedy_id_tree(tree_data, tree_classifiers, feature_test("tree_type"))
#print id_tree_classify_point(tree_test_point, tree_tree)

## To construct an ID tree for 2012 Q2 (Angels) or 2013 Q3 (numeric ID trees):
#print construct_greedy_id_tree(angel_data, angel_classifiers, feature_test("Classification"))
#print construct_greedy_id_tree(numeric_data, numeric_classifiers, feature_test("class"))


#### MULTIPLE CHOICE

ANSWER_1 = "bark_texture"
ANSWER_2 = "leaf_shape"
ANSWER_3 = "orange_foliage"

ANSWER_4 = [2, 3]
ANSWER_5 = [3]
ANSWER_6 = [2]
ANSWER_7 = 2

ANSWER_8 = "No"
ANSWER_9 = "No"


################################################################################
############################# k-NEAREST NEIGHBORS ##############################
################################################################################

#### MULTIPLE CHOICE: DRAWING BOUNDARIES

BOUNDARY_ANS_1 = 3
BOUNDARY_ANS_2 = 4

BOUNDARY_ANS_3 = 1
BOUNDARY_ANS_4 = 2

BOUNDARY_ANS_5 = 2
BOUNDARY_ANS_6 = 4
BOUNDARY_ANS_7 = 1
BOUNDARY_ANS_8 = 4
BOUNDARY_ANS_9 = 4

BOUNDARY_ANS_10 = 4
BOUNDARY_ANS_11 = 2
BOUNDARY_ANS_12 = 1
BOUNDARY_ANS_13 = 4
BOUNDARY_ANS_14 = 4


#### WARM-UP: DISTANCE METRICS

def dot_product(u, v):
    """Computes dot product of two vectors u and v, each represented as a tuple
    or list of coordinates.  Assume the two vectors are the same length."""
    sum = 0
    for i in range(len(u)):
        sum += u[i] * v[i]
    return sum

def norm(v):
    "Computes length of a vector v, represented as a tuple or list of coords."
    return dot_product(v, v) ** 0.5

def euclidean_distance(point1, point2):
    "Given two Points, computes and returns the Euclidean distance between them."
    sum = 0
    for i in range(len(point1.coords)):
        sum += (point1.coords[i] - point2.coords[i]) ** 2
    return sum ** 0.5

def manhattan_distance(point1, point2):
    "Given two Points, computes and returns the Manhattan distance between them."
    sum = 0
    for i in range(len(point1.coords)):
        sum += abs(point1.coords[i] - point2.coords[i])
    return sum

def hamming_distance(point1, point2):
    "Given two Points, computes and returns the Hamming distance between them."
    sum = 0
    for i in range(len(point1.coords)):
        sum += abs(point1.coords[i] != point2.coords[i])
    return sum

def cosine_distance(point1, point2):
    """Given two Points, computes and returns the cosine distance between them,
    where cosine distance is defined as 1-cos(angle_between(point1, point2))."""
    return 1 - dot_product(point1.coords, point2.coords) / (norm(point1.coords) * norm(point2.coords))


#### CLASSIFYING POINTS

def get_k_closest_points(point, data, k, distance_metric):
    """Given a test point, a list of points (the data), an int 0 < k <= len(data),
    and a distance metric (a function), returns a list containing the k points
    from the data that are closest to the test point, according to the distance
    metric.  Breaks ties lexicographically by coordinates."""
    return sorted(data, key=lambda i: (distance_metric(point, i), i.coords))[0:k]

def knn_classify_point(point, data, k, distance_metric):
    """Given a test point, a list of points (the data), an int 0 < k <= len(data),
    and a distance metric (a function), returns the classification of the test
    point based on its k nearest neighbors, as determined by the distance metric.
    Assumes there are no ties."""
    p = get_k_closest_points(point, data, k, distance_metric)
    l = []
    for i in p:
        l.append(i.classification)
    l = sorted(l, key=lambda t: l.count(t), reverse=True)
    return l[0]

## To run your classify function on the k-nearest neighbors problem from 2014 Q2
## part B2, uncomment the line below and try different values of k:
#print knn_classify_point(knn_tree_test_point, knn_tree_data, 5, euclidean_distance)


#### CHOOSING k

def cross_validate(data, k, distance_metric):
    """Given a list of points (the data), an int 0 < k <= len(data), and a
    distance metric (a function), performs leave-one-out cross-validation.
    Return the fraction of points classified correctly, as a float."""
    sum = 0
    for i in data:
        temp = []
        for j in data:
            if j != i:
                temp.append(j)
        if i.classification == knn_classify_point(i, temp, k, distance_metric):
            sum += 1
    return float(sum) / float(len(data))

def find_best_k_and_metric(data):
    """Given a list of points (the data), uses leave-one-out cross-validation to
    determine the best value of k and distance_metric, choosing from among the
    four distance metrics defined above.  Returns a tuple (k, distance_metric),
    where k is an int and distance_metric is a function."""
    max = 0
    result = None
    for k in range(1, len(data)):
        if cross_validate(data, k, euclidean_distance) > max:
            max = cross_validate(data, k, euclidean_distance)
            result = (k, euclidean_distance)
        if cross_validate(data, k, hamming_distance) > max:
            max = cross_validate(data, k, hamming_distance)
            result = (k, hamming_distance)
        if cross_validate(data, k, cosine_distance) > max:
            max = cross_validate(data, k, cosine_distance)
            result = (k, cosine_distance)
        if cross_validate(data, k, manhattan_distance) > max:
            max = cross_validate(data, k, manhattan_distance)
            result = (k, manhattan_distance)
    return  result

## To find the best k and distance metric for 2014 Q2, part B, uncomment:
#print find_best_k_and_metric(knn_tree_data)


#### MORE MULTIPLE CHOICE

kNN_ANSWER_1 = "Overfitting"
kNN_ANSWER_2 = "Underfitting"
kNN_ANSWER_3 = 4

kNN_ANSWER_4 = 4
kNN_ANSWER_5 = 1
kNN_ANSWER_6 = 3
kNN_ANSWER_7 = 3

#### SURVEY ###################################################

NAME = None
COLLABORATORS = None
HOW_MANY_HOURS_THIS_LAB_TOOK = None
WHAT_I_FOUND_INTERESTING = None
WHAT_I_FOUND_BORING = None
SUGGESTIONS = None
