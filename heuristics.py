from connect4 import *

occupation_eval = [[1,    1,    1,    1,    1,    1   ],
                   [1.25, 1.25, 1.25, 1.25, 1.25, 1.25],
                   [1.75, 1.75, 1.75, 1.75, 1.75, 1.75],
                   [2,    2,    2,    2,    2,    2   ],
                   [1.75, 1.75, 1.75, 1.75, 1.75, 1.75],
                   [1.25, 1.25, 1.25, 1.25, 1.25, 1.25],
                   [1,    1,    1,    1,    1,    1   ]]

def eval(board):
    e = 0
    for row in range(7):
        for col in range(6):
            e += occupation_eval[row][col] * board.board[row][col]
    return e