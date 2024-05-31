from connect4 import *
import torch
from torch import nn
import random
from copy import deepcopy

class NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fitness_score = None
        self.linear_relu_stack = nn.Sequential(
            nn.Linear((7*6) * 2, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, 2)
        )

    def forward(self, board):
        bt = torch.tensor(board.board)
        p1 = (bt + abs(bt)) / 2             # Board filtered for player 1 pieces
        p2 = abs(bt) - ((bt + abs(bt)) / 2) # Board filtered for player 2 pieces
        x = torch.flatten(torch.cat((p1, p2)))
        logits = self.linear_relu_stack(x)
        return logits
    
    def mutate(self, mut_range):
        for param in self.parameters():
            param.data += (torch.rand_like(param.data) - 0.5) * 2 * mut_range
    
    def minimax(self, board, depth, maximizing, alpha, beta):
        if board.is_draw():
            return [0, 0]
        else:
            winner = board.get_winner()
            if winner != Board.NONE:
                return [(1000000 + depth) * winner, 0]
        if depth == 0:
            eval = self.forward(board)
            return [eval[0] - eval[1], 0]
        
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

# Crosser over two models with the resulting model having a mix of parameters from both parent models
def crossover(model1: NN, model2: NN):
    result_model = NN()
    for param1, param2, res_param in zip(model1.parameters(), model2.parameters(), result_model.parameters()):
        shape = param1.data.shape
        if len(shape) == 1:
            for i in range(shape[0]):
                res_param.data[i] = random.choice([param1.data[i], param2.data[i]])
        elif len(shape) == 2:
            for i in range(shape[0]):
                for j in range(shape[1]):
                    res_param.data[i, j] = random.choice([param1.data[i, j], param2.data[i, j]])
        elif len(shape) == 3:
            for i in range(shape[0]):
                for j in range(shape[1]):
                    for k in range(shape[2]):
                        res_param.data[i, j, k] = random.choice([param1.data[i, j, k], param2.data[i, j, k]])
    return result_model
