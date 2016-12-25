# MIT 6.034 Lab 4: Constraint Satisfaction Problems
# Written by Dylan Holmes (dxh), Jessica Noss (jmn), and 6.034 staff

from constraint_api import *
from test_problems import get_pokemon_problem

#### PART 1: WRITE A DEPTH-FIRST SEARCH CONSTRAINT SOLVER

def has_empty_domains(csp) :
    "Returns True if the problem has one or more empty domains, otherwise False"
    for var in csp.domains:
        if csp.get_domain(var) == []:
            return True
    return False

def check_all_constraints(csp) :
    """Return False if the problem's assigned values violate some constraint,
    otherwise True"""
    values = csp.assigned_values
    for var1 in values:
        for var2 in values:
            for constraint in csp.constraints_between(var1, var2):
                if not constraint.check(values[var1], values[var2]):
                    return False
    return True

def solve_constraint_dfs(problem) :
    """Solves the problem using depth-first search.  Returns a tuple containing:
    1. the solution (a dictionary mapping variables to assigned values), and
    2. the number of extensions made (the number of problems popped off the agenda).
    If no solution was found, return None as the first element of the tuple."""
    num_extensions = 0
    agenda = [problem]
    while len(agenda) > 0:
        csp = agenda.pop(0)
        num_extensions += 1
        if (not has_empty_domains(csp)) and check_all_constraints(csp):
            if len(csp.unassigned_vars) == 0:
                return (csp.assigned_values, num_extensions)
            else:
                temp = []
                var = csp.pop_next_unassigned_var()
                for val in csp.get_domain(var):
                    newproblem = csp.copy()
                    newproblem.set_assigned_value(var, val)
                    temp.append(newproblem)
                agenda = temp + agenda
    return (None, num_extensions)



#### PART 2: DOMAIN REDUCTION BEFORE SEARCH

def eliminate_from_neighbors(csp, var) :
    """Eliminates incompatible values from var's neighbors' domains, modifying
    the original csp.  Returns an alphabetically sorted list of the neighboring
    variables whose domains were reduced, with each variable appearing at most
    once.  If no domains were reduced, returns empty list.
    If a domain is reduced to size 0, quits immediately and returns None."""
    domain = csp.get_domain(var)
    result = []
    for neighbor in csp.get_neighbors(var):
        eliminated = []
        ndomain = csp.get_domain(neighbor)
        flag = False
        for nval in ndomain:
            status = 0
            for val in domain:
                for constraint in csp.constraints_between(neighbor, var):
                    if not constraint.check(nval, val):
                        status += 1
                if status == len(domain):
                    eliminated.append(nval)
                    flag = True
        for i in eliminated:
            csp.eliminate(neighbor, i)
        if flag:
            result.append(neighbor)
        if len(csp.get_domain(neighbor)) == 0:
            return None
    return sorted(result)

def domain_reduction(csp, queue=None) :
    """Uses constraints to reduce domains, modifying the original csp.
    If queue is None, initializes propagation queue by adding all variables in
    their default order.  Returns a list of all variables that were dequeued,
    in the order they were removed from the queue.  Variables may appear in the
    list multiple times.
    If a domain is reduced to size 0, quits immediately and returns None."""
    if queue == None:
        queue = csp.get_all_variables()
    dequeue = []
    while len(queue) > 0:
        var = queue.pop(0)
        dequeue.append(var)
        eliminate = eliminate_from_neighbors(csp, var)
        if eliminate == None:
            return None
        else:
            queue = queue + eliminate
    return dequeue


# QUESTION 1: How many extensions does it take to solve the Pokemon problem
#    with dfs if you DON'T use domain reduction before solving it?

# Hint: Use get_pokemon_problem() to get a new copy of the Pokemon problem
#    each time you want to solve it with a different search method.

ANSWER_1 = solve_constraint_dfs(get_pokemon_problem())[1]

# QUESTION 2: How many extensions does it take to solve the Pokemon problem
#    with dfs if you DO use domain reduction before solving it?

pokeman = get_pokemon_problem()
domain_reduction(pokeman)
ANSWER_2 = solve_constraint_dfs(pokeman)[1]


#### PART 3: PROPAGATION THROUGH REDUCED DOMAINS

def solve_constraint_propagate_reduced_domains(problem) :
    """Solves the problem using depth-first search with forward checking and
    propagation through all reduced domains.  Same return type as
    solve_constraint_dfs."""
    num_extensions = 0
    agenda = [problem]
    while len(agenda) > 0:
        csp = agenda.pop(0)
        num_extensions += 1
        if (not has_empty_domains(csp)) and check_all_constraints(csp):
            if len(csp.unassigned_vars) == 0:
                return (csp.assigned_values, num_extensions)
            else:
                temp = []
                var = csp.pop_next_unassigned_var()
                for val in csp.get_domain(var):
                    newproblem = csp.copy()
                    newproblem.set_assigned_value(var, val)
                    queue = []
                    for assigned in newproblem.assigned_values:
                        queue.append(assigned)
                    domain_reduction(newproblem, queue)
                    temp.append(newproblem)
                agenda = temp + agenda
    return (None, num_extensions)

# QUESTION 3: How many extensions does it take to solve the Pokemon problem
#    with propagation through reduced domains? (Don't use domain reduction
#    before solving it.)

ANSWER_3 = solve_constraint_propagate_reduced_domains(get_pokemon_problem())[1]


#### PART 4: PROPAGATION THROUGH SINGLETON DOMAINS

def domain_reduction_singleton_domains(csp, queue=None) :
    """Uses constraints to reduce domains, modifying the original csp.
    Only propagates through singleton domains.
    Same return type as domain_reduction."""
    if queue == None:
        queue = csp.get_all_variables()
    dequeue = []
    while len(queue) > 0:
        var = queue.pop(0)
        dequeue.append(var)
        eliminate = eliminate_from_neighbors(csp, var)
        if eliminate == None:
            return None
        else:
            for i in eliminate:
                if len(csp.get_domain(i)) == 1:
                    queue.append(i)
    return dequeue

def solve_constraint_propagate_singleton_domains(problem) :
    """Solves the problem using depth-first search with forward checking and
    propagation through singleton domains.  Same return type as
    solve_constraint_dfs."""
    num_extensions = 0
    agenda = [problem]
    while len(agenda) > 0:
        csp = agenda.pop(0)
        num_extensions += 1
        if (not has_empty_domains(csp)) and check_all_constraints(csp):
            if len(csp.unassigned_vars) == 0:
                return (csp.assigned_values, num_extensions)
            else:
                temp = []
                var = csp.pop_next_unassigned_var()
                for val in csp.get_domain(var):
                    newproblem = csp.copy()
                    newproblem.set_assigned_value(var, val)
                    queue = []
                    for assigned in newproblem.assigned_values:
                        queue.append(assigned)
                    domain_reduction_singleton_domains(newproblem, queue)
                    temp.append(newproblem)
                agenda = temp + agenda
    return (None, num_extensions)

# QUESTION 4: How many extensions does it take to solve the Pokemon problem
#    with propagation through singleton domains? (Don't use domain reduction
#    before solving it.)

ANSWER_4 = solve_constraint_propagate_singleton_domains(get_pokemon_problem())[1]


#### PART 5: FORWARD CHECKING

def propagate(enqueue_condition_fn, csp, queue=None) :
    """Uses constraints to reduce domains, modifying the original csp.
    Uses enqueue_condition_fn to determine whether to enqueue a variable whose
    domain has been reduced.  Same return type as domain_reduction."""
    if queue == None:
        queue = csp.get_all_variables()
    dequeue = []
    while len(queue) > 0:
        var = queue.pop(0)
        dequeue.append(var)
        eliminate = eliminate_from_neighbors(csp, var)
        if eliminate == None:
            return None
        else:
            for i in eliminate:
                if enqueue_condition_fn(csp, i):
                    queue.append(i)
    return dequeue

def condition_domain_reduction(csp, var) :
    """Returns True if var should be enqueued under the all-reduced-domains
    condition, otherwise False"""
    return True

def condition_singleton(csp, var) :
    """Returns True if var should be enqueued under the singleton-domains
    condition, otherwise False"""
    return len(csp.get_domain(var)) == 1

def condition_forward_checking(csp, var) :
    """Returns True if var should be enqueued under the forward-checking
    condition, otherwise False"""
    return False


#### PART 6: GENERIC CSP SOLVER

def solve_constraint_generic(problem, enqueue_condition=None) :
    """Solves the problem, calling propagate with the specified enqueue
    condition (a function).  If enqueue_condition is None, uses DFS only.
    Same return type as solve_constraint_dfs."""
    num_extensions = 0
    agenda = [problem]
    while len(agenda) > 0:
        csp = agenda.pop(0)
        num_extensions += 1
        if (not has_empty_domains(csp)) and check_all_constraints(csp):
            if len(csp.unassigned_vars) == 0:
                return (csp.assigned_values, num_extensions)
            else:
                temp = []
                var = csp.pop_next_unassigned_var()
                for val in csp.get_domain(var):
                    newproblem = csp.copy()
                    newproblem.set_assigned_value(var, val)
                    if enqueue_condition != None:
                        propagate(enqueue_condition, newproblem, [var])
                    temp.append(newproblem)
                agenda = temp + agenda
    return (None, num_extensions)

# QUESTION 5: How many extensions does it take to solve the Pokemon problem
#    with DFS and forward checking, but no propagation? (Don't use domain
#    reduction before solving it.)

ANSWER_5 = solve_constraint_generic(get_pokemon_problem(), condition_forward_checking)[1]


#### PART 7: DEFINING CUSTOM CONSTRAINTS

def constraint_adjacent(m, n) :
    """Returns True if m and n are adjacent, otherwise False.
    Assume m and n are ints."""
    return abs(m - n) == 1

def constraint_not_adjacent(m, n) :
    """Returns True if m and n are NOT adjacent, otherwise False.
    Assume m and n are ints."""
    return not constraint_adjacent(m, n)

def all_different(variables) :
    """Returns a list of constraints, with one difference constraint between
    each pair of variables."""
    result = []
    for i in variables:
        for j in variables:
            if i > j:
                result.append(Constraint(i, j, constraint_different))
    return result

#### PART 8: MOOSE PROBLEM (OPTIONAL)

moose_problem = ConstraintSatisfactionProblem(["You", "Moose", "McCain",
                                               "Palin", "Obama", "Biden"])

# Add domains and constraints to your moose_problem here:


# To test your moose_problem AFTER implementing all the solve_constraint
# methods above, change TEST_MOOSE_PROBLEM to True:
TEST_MOOSE_PROBLEM = False


#### SURVEY ###################################################

NAME = "Yifan Wang"
COLLABORATORS = ""
HOW_MANY_HOURS_THIS_LAB_TOOK = 2
WHAT_I_FOUND_INTERESTING = "Generic function"
WHAT_I_FOUND_BORING = ""
SUGGESTIONS = ""


###########################################################
### Ignore everything below this line; for testing only ###
###########################################################

if TEST_MOOSE_PROBLEM:
    # These lines are used in the local tester iff TEST_MOOSE_PROBLEM is True
    moose_answer_dfs = solve_constraint_dfs(moose_problem.copy())
    moose_answer_propany = solve_constraint_propagate_reduced_domains(moose_problem.copy())
    moose_answer_prop1 = solve_constraint_propagate_singleton_domains(moose_problem.copy())
    moose_answer_generic_dfs = solve_constraint_generic(moose_problem.copy(), None)
    moose_answer_generic_propany = solve_constraint_generic(moose_problem.copy(), condition_domain_reduction)
    moose_answer_generic_prop1 = solve_constraint_generic(moose_problem.copy(), condition_singleton)
    moose_answer_generic_fc = solve_constraint_generic(moose_problem.copy(), condition_forward_checking)
    moose_instance_for_domain_reduction = moose_problem.copy()
    moose_answer_domain_reduction = domain_reduction(moose_instance_for_domain_reduction)
    moose_instance_for_domain_reduction_singleton = moose_problem.copy()
    moose_answer_domain_reduction_singleton = domain_reduction_singleton_domains(moose_instance_for_domain_reduction_singleton)
