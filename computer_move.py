from connect4 import *
from copy import deepcopy

evlauation_function = None

def set_evaluation_function(func):
    global evlauation_function
    evlauation_function = func

def minimax(board, depth, maximizing, alpha, beta):
    if board.is_draw():
        return [0, 0]
    else:
        winner = board.get_winner()
        if winner != Board.NONE:
            return [(1000000 + depth) * winner, 0]
    if depth == 0:
        return [evlauation_function(board), 0]
    
    if maximizing:
        best_val = float("-inf")
        best_move = 0
        for col in board.get_free_columns():
            hypothetical_board = deepcopy(board)
            hypothetical_board.place(col, Board.X)
            value = minimax(hypothetical_board, depth - 1, False, alpha, beta)[0]
            if value > best_val:
                best_val = value
                best_move = col
            alpha = max(alpha, best_val)
            if beta <= alpha:
                break
    else:
        best_val = float("inf")
        best_move = 0
        for col in board.get_free_columns():
            hypothetical_board = deepcopy(board)
            hypothetical_board.place(col, Board.O)
            value = minimax(hypothetical_board, depth - 1, True, alpha, beta)[0]
            if value < best_val:
                best_val = value
                best_move = col
            beta = min(beta, best_val)
            if beta <= alpha:
                break
    return [best_val, best_move]