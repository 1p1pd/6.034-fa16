# MIT 6.034 Lab 3: Games
# Written by Dylan Holmes (dxh), Jessica Noss (jmn), and 6.034 staff

from game_api import *
from boards import *
INF = float('inf')

def is_game_over_connectfour(board) :
    "Returns True if game is over, otherwise False."
    if board.count_pieces() == 42:
        return True
    for chain in board.get_all_chains():
        if len(chain) >= 4:
            return True
    return False

def next_boards_connectfour(board) :
    result = []
    if is_game_over_connectfour(board):
        return result
    for i in range(board.num_cols):
        if not board.is_column_full(i):
            result.append(board.add_piece(i))
    return result

def endgame_score_connectfour(board, is_current_player_maximizer) :
    """Given an endgame board, returns 1000 if the maximizer has won,
    -1000 if the minimizer has won, or 0 in case of a tie."""
    status = False
    for chain in board.get_all_chains():
        if len(chain) >= 4:
            status = True
    if not status:
        return 0
    if is_current_player_maximizer:
        return -1000
    else:
        return 1000

def endgame_score_connectfour_faster(board, is_current_player_maximizer) :
    """Given an endgame board, returns an endgame score with abs(score) >= 1000,
    returning larger absolute scores for winning sooner."""
    status = False
    for chain in board.get_all_chains():
        if len(chain) >= 4:
            status = True
    if not status:
        return 0
    if is_current_player_maximizer:
        return -1000 - 42 + board.count_pieces()
    else:
        return 1000 + 42 - board.count_pieces()

def heuristic_connectfour(board, is_current_player_maximizer) :
    """Given a non-endgame board, returns a heuristic score with
    abs(score) < 1000, where higher numbers indicate that the board is better
    for the maximizer."""
    firstchain = 0
    for chain in board.get_all_chains(is_current_player_maximizer):
        if len(chain) == 2:
            firstchain += 10
        elif len(chain) == 3:
            firstchain += 100
    secondchain = 0
    for chain in board.get_all_chains(not is_current_player_maximizer):
        if len(chain) == 2:
            secondchain += 10
        elif len(chain) == 3:
            secondchain += 100
    return firstchain - secondchain

# Now we can create AbstractGameState objects for Connect Four, using some of
# the functions you implemented above.  You can use the following examples to
# test your dfs and minimax implementations in Part 2.

# This AbstractGameState represents a new ConnectFourBoard, before the game has started:
state_starting_connectfour = AbstractGameState(snapshot = ConnectFourBoard(),
                                 is_game_over_fn = is_game_over_connectfour,
                                 generate_next_states_fn = next_boards_connectfour,
                                 endgame_score_fn = endgame_score_connectfour_faster)

# This AbstractGameState represents the ConnectFourBoard "NEARLY_OVER" from boards.py:
state_NEARLY_OVER = AbstractGameState(snapshot = NEARLY_OVER,
                                 is_game_over_fn = is_game_over_connectfour,
                                 generate_next_states_fn = next_boards_connectfour,
                                 endgame_score_fn = endgame_score_connectfour_faster)

# This AbstractGameState represents the ConnectFourBoard "BOARD_UHOH" from boards.py:
state_UHOH = AbstractGameState(snapshot = BOARD_UHOH,
                                 is_game_over_fn = is_game_over_connectfour,
                                 generate_next_states_fn = next_boards_connectfour,
                                 endgame_score_fn = endgame_score_connectfour_faster)


#### PART 2 ###########################################
# Note: Functions in Part 2 use the AbstractGameState API, not ConnectFourBoard.

def dfs_maximizing(state) :
    """Performs depth-first search to find path with highest endgame score.
    Returns a tuple containing:
     0. the best path (a list of AbstractGameState objects),
     1. the score of the leaf node (a number), and
     2. the number of static evaluations performed (a number)"""
    if state.is_game_over():
        return ([state], state.get_endgame_score(), 1)
    else:
        num = 0
        max = -INF
        best = []
        for nextstate in state.generate_next_states():
            result = dfs_maximizing(nextstate)
            num += result[2]
            if max < result[1]:
                max = result[1]
                best = [state]
                for path in result[0]:
                    best.append(path)
        return (best, max, num)


def minimax_endgame_search(state, maximize=True) :
    """Performs minimax search, searching all leaf nodes and statically
    evaluating all endgame scores.  Same return type as dfs_maximizing."""
    if state.is_game_over():
        return ([state], state.get_endgame_score(maximize), 1)
    else:
        num = 0
        max = -INF
        min = INF
        best = []
        for nextstate in state.generate_next_states():
            result = minimax_endgame_search(nextstate, not maximize)
            num += result[2]
            if maximize:
                if max < result[1]:
                    max = result[1]
                    best = [state]
                    for path in result[0]:
                        best.append(path)
            else:
                if min > result[1]:
                    min = result[1]
                    best = [state]
                    for path in result[0]:
                        best.append(path)
        if maximize:
            return (best, max, num)
        else:
            return (best, min, num)

# Uncomment the line below to try your minimax_endgame_search on an
# AbstractGameState representing the ConnectFourBoard "NEARLY_OVER" from boards.py:

#pretty_print_dfs_type(minimax_endgame_search(state_NEARLY_OVER))


def minimax_search(state, heuristic_fn=always_zero, depth_limit=INF, maximize=True) :
    "Performs standard minimax search.  Same return type as dfs_maximizing."
    if state.is_game_over():
        return ([state], state.get_endgame_score(maximize), 1)
    elif depth_limit == 0:
        return ([state], heuristic_fn(state.get_snapshot(), maximize), 1)
    else:
        num = 0
        max = -INF
        min = INF
        best = []
        for nextstate in state.generate_next_states():
            result = minimax_search(nextstate, heuristic_fn, depth_limit - 1, not maximize)
            num += result[2]
            if maximize:
                if max < result[1]:
                    max = result[1]
                    best = [state]
                    for path in result[0]:
                        best.append(path)
            else:
                if min > result[1]:
                    min = result[1]
                    best = [state]
                    for path in result[0]:
                        best.append(path)
        if maximize:
            return (best, max, num)
        else:
            return (best, min, num)

# Uncomment the line below to try minimax_search with "BOARD_UHOH" and
# depth_limit=1.  Try increasing the value of depth_limit to see what happens:

#pretty_print_dfs_type(minimax_search(state_UHOH, heuristic_fn=heuristic_connectfour, depth_limit=1))


def minimax_search_alphabeta(state, alpha=-INF, beta=INF, heuristic_fn=always_zero,
                             depth_limit=INF, maximize=True) :
    "Performs minimax with alpha-beta pruning.  Same return type as dfs_maximizing."
    if state.is_game_over():
        return ([state], state.get_endgame_score(maximize), 1)
    elif depth_limit == 0:
        return ([state], heuristic_fn(state.get_snapshot(), maximize), 1)
    else:
        num = 0
        best = []
        for nextstate in state.generate_next_states():
            result = minimax_search_alphabeta(nextstate, alpha, beta, heuristic_fn, depth_limit - 1, not maximize)
            num += result[2]
            if maximize:
                if alpha < result[1]:
                    alpha = result[1]
                    best = [state]
                    for path in result[0]:
                        best.append(path)
                if alpha >= beta:
                    break
            else:
                if beta > result[1]:
                    beta = result[1]
                    best = [state]
                    for path in result[0]:
                        best.append(path)
                if alpha >= beta:
                    break
        if maximize:
            return (best, alpha, num)
        else:
            return (best, beta, num)

# Uncomment the line below to try minimax_search_alphabeta with "BOARD_UHOH" and
# depth_limit=4.  Compare with the number of evaluations from minimax_search for
# different values of depth_limit.

#pretty_print_dfs_type(minimax_search_alphabeta(state_UHOH, heuristic_fn=heuristic_connectfour, depth_limit=4))


def progressive_deepening(state, heuristic_fn=always_zero, depth_limit=INF,
                          maximize=True) :
    """Runs minimax with alpha-beta pruning. At each level, updates anytime_value
    with the tuple returned from minimax_search_alphabeta. Returns anytime_value."""
    anytime_value = AnytimeValue()   # TA Note: Use this to store values.
    i = 1
    while i <= depth_limit:
        anytime_value.set_value(minimax_search_alphabeta(state, -INF, INF, heuristic_fn, i, maximize))
        i += 1
    return anytime_value

# Uncomment the line below to try progressive_deepening with "BOARD_UHOH" and
# depth_limit=4.  Compare the total number of evaluations with the number of
# evaluations from minimax_search or minimax_search_alphabeta.

#print progressive_deepening(state_UHOH, heuristic_fn=heuristic_connectfour, depth_limit=4)


##### PART 3: Multiple Choice ##################################################

ANSWER_1 = '4'

ANSWER_2 = '1'

ANSWER_3 = '4'

ANSWER_4 = '5'


#### SURVEY ###################################################

NAME = "Yifan Wang"
COLLABORATORS = ""
HOW_MANY_HOURS_THIS_LAB_TOOK = 3
WHAT_I_FOUND_INTERESTING = "everything"
WHAT_I_FOUND_BORING = "None"
SUGGESTIONS = None
