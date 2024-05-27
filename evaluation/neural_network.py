from connect4 import *
import torch
from torch import nn
import random
from copy import deepcopy

class NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_lrelu_stack = nn.Sequential(
            nn.Linear(42, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, board):
        x = self.flatten(torch.tensor([board.board]).to(torch.float32))
        logits = self.linear_lrelu_stack(x)
        return logits
    
    def minimax(self, board, depth, maximizing, alpha, beta):
        if board.is_draw():
            return [0, 0]
        else:
            winner = board.get_winner()
            if winner != Board.NONE:
                return [(1000000 + depth) * winner, 0]
        if depth == 0:
            return [self.forward(board).tolist()[0][0], 0]
        
        if maximizing:
            best_val = float("-inf")
            best_move = 0
            for col in board.get_free_columns():
                hypothetical_board = deepcopy(board)
                hypothetical_board.place(col, Board.X)
                value = self.minimax(hypothetical_board, depth - 1, False, alpha, beta)[0]
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
                value = self.minimax(hypothetical_board, depth - 1, True, alpha, beta)[0]
                if value < best_val:
                    best_val = value
                    best_move = col
                beta = min(beta, best_val)
                if beta <= alpha:
                    break
        return [best_val, best_move]

# "Breeds" 2 models by randomly combining their parameters, as well as adding random mutations
def breed(model1, model2, mutation):
    result_model = NN()
    # Combine parameters randomly from both parents
    for param1, param2, param3 in zip(model1.parameters(), model2.parameters(), result_model.parameters()):
        param3.data.copy_(random.choice([param1.data, param2.data]))
        param3.data.copy_(random.choice([param1.data, param2.data]))
    # Introduce mutations
    for param in result_model.parameters():
        param.data += torch.rand_like(param.data) * (-2 * mutation) + mutation
    return result_model
