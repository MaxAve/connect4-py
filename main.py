from connect4 import *
import os
import computer_move
import torch
import random
from datetime import datetime
from models_v2 import neural_network

# Plays a game between the given AI and a preprogrammed heristic model
# Returns a result between -1 and 1 based on how well the model performed
# Below 0 = loss (lower = lost in less moves)
# 0 = draw
# Above 0 = win (larger = won in less moves)
def model_vs_heuristic(model, game_id, model_plays_as_x=True):
    print(f"Starting game {game_id}...")
    board = Board()
    moves_played = 0
    try:
        if model_plays_as_x:
            while board.get_winner() == Board.NONE and board.is_draw() == False:
                # Computer move (neural network)
                best_move = model.minimax(board, 4, True, float("-inf"), float("inf"))
                board.place(best_move[1], Board.X)
                if not (board.get_winner() == Board.NONE and board.is_draw() == False):
                    break
                # Computer move (heuristics)
                best_move = computer_move.minimax(board, 4, False, float("-inf"), float("inf"))
                board.place(best_move[1], Board.O)
                moves_played += 1
        else:
            while board.get_winner() == Board.NONE and board.is_draw() == False:
                # Computer move (heuristics)
                best_move = computer_move.minimax(board, 4, True, float("-inf"), float("inf"))
                board.place(best_move[1], Board.X)
                if not (board.get_winner() == Board.NONE and board.is_draw() == False):
                    break
                # Computer move (neural network)
                best_move = model.minimax(board, 4, False, float("-inf"), float("inf"))
                board.place(best_move[1], Board.O)
                moves_played += 1
    except Exception as e:
        print(f"Game {game_id} exited with an error: {e}")
        return -1
    score = 0
    winner = board.get_winner()
    if model_plays_as_x:
        if winner == Board.X:
            score = (42 - moves_played) / 42
        else:
            score = (moves_played - 42) / 42
    else:
        if winner == Board.O:
            score = (42 - moves_played) / 42
        else:
            score = (moves_played - 42) / 42
    if board.is_draw():
        score = 0
    print(f"Finished game {game_id} with a score of {score}.")
    return score


def train(generations, survivor_population_size, mutation, name, sample=None):
    offspring_population_size = survivor_population_size * (survivor_population_size - 1)
    total_generation_size = survivor_population_size + offspring_population_size

    population = []
    # Initialize the starting population
    if sample is None:
        for _ in range(total_generation_size):
            population.append(neural_network.NN())
    else:
        for i in range(total_generation_size):
            nn = neural_network.NN()
            # Give mutations to all agents except for the first one
            if i > 0:
                nn.mutate(mutation)
            population.append(nn)

    # Train
    for g in range(generations):
        print(f"Spawning generation {g}...")
        # Let every model play a game against a preprogrammed heuristic model
        for i in range(len(population)):
            population[i].fitness_score = model_vs_heuristic(population[i], i, random.choice([True, False]))

        fitness_score_sum = 0
        for model in population:
            fitness_score_sum += model.fitness_score
        fitness_score_sum /= len(population)
        print(f"Average fitness score: {round(fitness_score_sum, 2)}")

        # Find the best models
        best_models = []
        while len(best_models) < survivor_population_size:
            max_score = -1
            best_model_index = 0
            for i in range(len(population)):
                if population[i].fitness_score is None:
                    print("Skipping model due to lack of fitness score...")
                    continue
                if population[i].fitness_score > max_score:
                    max_score = population[i].fitness_score
                    best_model_index = i
            print(f"Model with a fitness score of {population[best_model_index].fitness_score} will be used for crossover.")
            best_models.append(population[best_model_index])
            del population[best_model_index]

        population.clear()

        # Breed the best models
        for a in range(len(best_models)):
            for b in range(len(best_models)):
                if a != b:
                    offspring = neural_network.crossover(best_models[a], best_models[b])
                    offspring.mutate(mutation)
                    population.append(offspring)
        # Elitism (keep winners in the next generation to avoid accidental devolution)
        for model in best_models:
            population.append(model)

    # Save the model that performed the best
    max_score = -1
    winner_model = None
    for model in best_models:
        if model.fitness_score > max_score:
            max_score = model.fitness_score
            winner_model = model
    print(f"Saving model as {name}...")
    torch.save(winner_model.state_dict(), f"/home/maxave/Desktop/connect4-py/models_v2/{name}")


def play_against(model):
    board = Board()
    while board.get_winner() == Board.NONE and board.is_draw() == False:
        os.system("clear")
        print(board)

        # Computer move (neural network)
        best_move = model.minimax(board, 6, True, float("-inf"), float("inf"))
        board.place(best_move[1], Board.X)

        if not (board.get_winner() == Board.NONE and board.is_draw() == False):
            break

        os.system("clear")
        print(board)

        # Player move
        col = int(input("Your move (column): "))
        col = min(7, max(1, col))
        board.place(col - 1, Board.O)
    os.system("clear")
    print(board)
    print("YOU LOST!" if board.get_winner() == Board.X else ("YOU WIN!" if board.get_winner() == Board.O else "IT'S A DRAW!"))

def main():
    model = neural_network.NN()
    model.load_state_dict(torch.load("/home/maxave/Desktop/connect4-py/models_v2/connect4-eval-05-31-2024_13:07:17"))
    model.eval()
    play_against(model=model)
    # train(10, 5, 0.1, f"connect4-eval-{datetime.now().strftime('%m-%d-%Y_%H:%M:%S')}")

if __name__ == "__main__":
    main()
