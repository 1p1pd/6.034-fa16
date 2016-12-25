# MIT 6.034 Lab 8: Bayesian Inference
# Written by Dylan Holmes (dxh), Jessica Noss (jmn), and 6.034 staff

from nets import *


#### ANCESTORS, DESCENDANTS, AND NON-DESCENDANTS ###############################

def get_ancestors(net, var):
    "Return a set containing the ancestors of var"
    result = set()
    if net.get_parents(var) != set([]):
        for p in net.get_parents(var):
            result.add(p)
            result = result.union(get_ancestors(net, p))
    return result

def get_descendants(net, var):
    "Returns a set containing the descendants of var"
    result = set()
    if net.get_children(var) != set([]):
        for p in net.get_children(var):
            result.add(p)
            result = result.union(get_descendants(net, p))
    return result

def get_nondescendants(net, var):
    "Returns a set containing the non-descendants of var"
    result = set()
    toposort = net.topological_sort()
    for i in toposort:
        if toposort.index(i) < toposort.index(var):
            result.add(i)
        elif toposort.index(i) > toposort.index(var):
            if var in net.get_parents(i):
                continue
            for p in net.get_parents(i):
                if (p in result or net.get_parents(i) == set([])) and i not in result:
                    result.add(i)
    return result


def simplify_givens(net, var, givens):
    """If givens include every parent of var and no descendants, returns a
    simplified list of givens, keeping only parents.  Does not modify original
    givens.  Otherwise, if not all parents are given, or if a descendant is
    given, returns original givens."""
    parents = net.get_parents(var)
    nondescend = get_nondescendants(net, var)
    for p in parents:
        if p not in givens:
            return givens
    for g in givens:
        if g not in nondescend:
            return givens
    result = {}
    for g in givens:
        if g in parents:
            result[g] = givens[g]
    return result


#### PROBABILITY ###############################################################

def probability_lookup(net, hypothesis, givens=None):
    "Looks up a probability in the Bayes net, or raises LookupError"
    var = None
    for i in hypothesis:
        var = i
    if givens is not None:
        givens = simplify_givens(net, var, givens)
    try:
        return net.get_probability(hypothesis, givens)
    except ValueError:
        raise LookupError

def probability_joint(net, hypothesis):
    "Uses the chain rule to compute a joint probability"
    result = 1
    topo = net.topological_sort()
    topo.reverse()
    for i in topo:
        if i in hypothesis:
            temp = {}
            temp[i] = hypothesis[i]
            del hypothesis[i]
            result *= probability_lookup(net, temp, hypothesis)
    return result

def probability_marginal(net, hypothesis):
    "Computes a marginal probability as a sum of joint probabilities"
    result = 0
    var = net.get_variables()
    for i in hypothesis:
        var.remove(i)
    for p in net.combinations(var, hypothesis):
        result += probability_joint(net, p)
    return result

def probability_conditional(net, hypothesis, givens=None):
    "Computes a conditional probability as a ratio of marginal probabilities"
    if givens is None:
        return probability_marginal(net, hypothesis)
    temp = dict(hypothesis, **givens)
    for i in hypothesis:
        if i in givens:
            if hypothesis[i] != givens[i]:
                return 0
    return probability_marginal(net, temp) / probability_marginal(net, givens)

def probability(net, hypothesis, givens=None):
    "Calls previous functions to compute any probability"
    return probability_conditional(net, hypothesis, givens)


#### PARAMETER-COUNTING AND INDEPENDENCE #######################################

def number_of_parameters(net):
    "Computes minimum number of parameters required for net"
    result = 0
    for i in net.topological_sort():
        if net.get_parents(i) == set([]):
            result += len(net.get_domain(i)) - 1
        else:
            temp = 1
            for p in net.get_parents(i):
                temp *= len(net.get_domain(p))
            temp *= len(net.get_domain(i)) - 1
            result += temp
    return result

def is_independent(net, var1, var2, givens=None):
    """Return True if var1, var2 are conditionally independent given givens,
    otherwise False.  Uses numerical independence."""
    v1 = net.combinations(var1)
    v2 = net.combinations(var2)
    for p1 in v1:
        for p2 in v2:
            p3 = dict(p1, **p2)
            if not approx_equal(probability_conditional(net, p1, givens) * probability_conditional(net, p2, givens),
                        probability_conditional(net, p3, givens)):
                return False
    return True

def is_structurally_independent(net, var1, var2, givens=None):
    """Return True if var1, var2 are conditionally independent given givens,
    based on the structure of the Bayes net, otherwise False.
    Uses structural independence only (not numerical independence)."""
    ancestral = set()
    if givens is not None:
        for i in givens:
            ancestral.update(get_ancestors(net, i))
            ancestral.update(i)
    ancestral.update(get_ancestors(net, var1))
    ancestral.update(var1)
    ancestral.update(get_ancestors(net, var2))
    ancestral.update(var2)
    ancent = net.subnet(ancestral)
    for v1 in ancent.get_variables():
        for v2 in ancent.get_variables():
            if v1 != v2:
                for c in ancent.get_children(v1):
                    if c in ancent.get_children(v2):
                        ancent = ancent.link(v1, v2)
    ancent = ancent.make_bidirectional()
    if givens is not None:
        for g in givens:
            ancent = ancent.remove_variable(g)
    return ancent.find_path(var1, var2) is None

#### SURVEY ####################################################################

NAME = "Yifan Wang"
COLLABORATORS = ""
HOW_MANY_HOURS_THIS_LAB_TOOK = 1
WHAT_I_FOUND_INTERESTING = "Everything"
WHAT_I_FOUND_BORING = ""
SUGGESTIONS = ""
