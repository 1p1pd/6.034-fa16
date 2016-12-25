# MIT 6.034 Lab 0: Getting Started
# Written by Jessica Noss (jmn), Dylan Holmes (dxh), and past 6.034 staff

from point_api import Point

# This is a multiple choice question. You answer by replacing
# the symbol 'None' with a letter, corresponding to your answer.

# What version of Python do we *recommend* (not "require") for this course?
#   A. Python v2.3
#   B. Python v2.5, Python v2.6, or Python v2.7
#   C. Python v3.0
# Fill in your answer in the next line of code ("A", "B", or "C"):

ANSWER_1 = "B"


#### Warm-up: Exponentiation ###################################################

def cube(x):
    return x**3


#### Recursion #################################################################

def fibonacci(n):
    if n < 0:
        raise ValueError("fibonacci: input must not be negative")
    elif n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci(n-1) + fibonacci(n-2)

def expression_depth(expr):
    if isinstance(expr, list):
        max = 0
        for elem in expr:
            if max < 1 + expression_depth(elem):
                max = 1 + expression_depth(elem)
        return max
    else:
        return 0


#### Built-in data types #######################################################

def compute_string_properties(string):
    if string == "":
        return (len(string), [], 0)
    else:
        result = compute_string_properties(string[1:])
        if result[1] == []:
            newList = [string[0]]
            return (len(string), newList, result[2]+1)
        elif string[0] not in result[1]:
            result[1].append(string[0])
            result[1].sort(reverse=True)
            return (len(string), result[1], result[2]+1)
        else:
            result[1].append(string[0])
            result[1].sort(reverse=True)
            return (len(string), result[1], result[2])


def tally_letters(string):
    dic = {}
    for i in range(len(string)):
        try:
            dic[string[i]] += 1
        except:
            dic[string[i]] = 1
    return dic


#### Functions that return functions ###########################################

def create_multiplier_function(m):
    def multiplies(n):
        return n*m
    return multiplies


#### Objects and APIs: Copying and modifing objects ##########################

def get_neighbors(point):
    left = point.copy()
    right = point.copy()
    up = point.copy()
    down = point.copy()
    x = point.getX()
    y = point.getY()
    left.setX(x-1)
    right.setX(x+1)
    up.setY(y+1)
    down.setY(y-1)
    return [left, right, down, up]


#### Using the "key" argument ##################################################

def sort_points_by_Y(list_of_points):
    Yaxis = lambda p: p.getY()
    result = sorted(list_of_points, key=Yaxis)
    return result

def furthest_right_point(list_of_points):
    Xaxis = lambda p: p.getX()
    result = max(list_of_points, key=Xaxis)
    return result


#### SURVEY ####################################################################

# How much programming experience do you have, in any language?
#     A. No experience (never programmed before this semester)
#     B. Beginner (just started learning to program, e.g. took one programming class)
#     C. Intermediate (have written programs for a couple classes/projects)
#     D. Proficient (have been programming for multiple years, or wrote programs for many classes/projects)
#     E. Expert (could teach a class on programming, either in a specific language or in general)

PROGRAMMING_EXPERIENCE = "D"  #type a letter (A, B, C, D, E) between the quotes


# How much experience do you have with Python?
#     A. No experience (never used Python before this semester)
#     B. Beginner (just started learning, e.g. took 6.0001)
#     C. Intermediate (have used Python in a couple classes/projects)
#     D. Proficient (have used Python for multiple years or in many classes/projects)
#     E. Expert (could teach a class on Python)

PYTHON_EXPERIENCE = "D"


# Finally, the following questions will appear at the end of every lab.
# The first three are required in order to receive full credit for your lab.

NAME = "Yifan Wang"
COLLABORATORS = ""
HOW_MANY_HOURS_THIS_LAB_TOOK = 1
SUGGESTIONS = None #optional
