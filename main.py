from connect4 import *
import computer_move
import heuristics
import neural_network
import os
from copy import deepcopy
import torch

# Plays a game between the given model and a preprogrammed heuristic model
# Use this during initial training stages
def train_with_heuristics(model, score_tally, game_id):
    board = Board()
    computer_move.set_evaluation_function(heuristics.eval)
    moves_played = 0

    try:
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
        print(board)
        print(board.get_winner())
    except:
        print(f"Game No. {game_id} exited with an error.")
        score_tally[game_id] = -1
        return

    # Calculate the score
    # This will save a number between -1 and 1 (-1/1 = lost/won game before it even started)
    if board.is_draw():
        score = 0
    elif board.get_winner() == Board.X:
        # Reduce penalty for games that lasted longer
        score = (moves_played - 42) / 42
    elif board.get_winner() == Board.O:
        # Reduce score for games that lasted longer
        score = (42 - moves_played) / 42
    score_tally[game_id] = score
    print(f"Finished game No. {game_id}. The model had an overall playing strength of {score} ({int(score*100)}%).")

# Has 2 models play a round against eachother
def train_with_model(model1, model2, score_tally, game_id):
    board = Board()
    computer_move.set_evaluation_function(heuristics.eval)
    moves_played = 0

    try:
        while board.get_winner() == Board.NONE and board.is_draw() == False:
            # Computer move (heuristics)
            best_move = model1.minimax(board, 4, False, float("-inf"), float("inf"))
            board.place(best_move[1], Board.X)
            if not (board.get_winner() == Board.NONE and board.is_draw() == False):
                break
            # Computer move (neural network)
            best_move = model2.minimax(board, 4, False, float("-inf"), float("inf"))
            board.place(best_move[1], Board.O)

            moves_played += 1
        print(board)
        print(board.get_winner())
    except:
        print(f"Game No. {game_id} exited with an error.")
        score_tally[game_id] = -1
        return

    # Calculate the score
    # This will save a number between -1 and 1 (-1/1 = lost/won game before it even started)
    if board.is_draw():
        score = 0
    elif board.get_winner() == Board.X:
        # Reduce penalty for games that lasted longer
        score = (moves_played - 42) / 42
    elif board.get_winner() == Board.O:
        # Reduce score for games that lasted longer
        score = (42 - moves_played) / 42
    score_tally[game_id] = score
    print(f"Finished game No. {game_id}. The model had an overall playing strength of {score} ({int(score*100)}%).")

def main():
    board = Board()
    '''
    breedable_population_size = 5 # Amount of models that will be allowed to breed after each round
    offspring_population_size = int(breedable_population_size * (breedable_population_size - 1))

    # Create initial population
    active_models = []
    model = neural_network.NN()
    model.load_state_dict(torch.load("models/nn-3"))
    model.eval()
    for _ in range(offspring_population_size):
        active_models.append(model)

    winner_model = active_models[0]

    for r in range(20):
        print(f"Starting round {r+1}.")

        score_tally = [0] * len(active_models)

        # Let every model play 1 round each with the heuristics bot
        # for i in range(len(active_models)):
        #     train_with_heuristics(active_models[i], score_tally, i)

        for i in range(len(active_models)):
            train_with_model(active_models[i], winner_model, score_tally, i)

        # Pick out the best ones
        best_models = []
        score_tally_tmp = deepcopy(score_tally)
        while len(best_models) < breedable_population_size:
            max_score = float("-inf")
            best_model_index = 0
            for i in range(len(active_models)):
                if score_tally_tmp[i] > max_score:
                    max_score = score_tally_tmp[i]
                    best_model_index = i
            if len(best_models) == 0:
                winner_model = active_models[best_model_index]
            best_models.append(active_models[best_model_index])
            del active_models[best_model_index]
            del score_tally_tmp[best_model_index]
        active_models.clear()

        # Calculate average
        score_sum = 0
        for score in score_tally:
            score_sum += score
        print(f"Average playing strength: {score_sum / float(len(score_tally))} ({int(score_sum / float(len(score_tally))*100)}%)")
        print(f"Best: {max(score_tally)} ({int(max(score_tally)*100)}%)")

        # Breed everybody with everybody else
        print("Breeding...")
        for i in range(len(best_models)):
            for j in range(len(best_models)):
                if i != j:
                    active_models.append(neural_network.breed(best_models[i], best_models[j], 0.05))
        #active_models += best_models # We want to keep using the winners from the last generation to prevent accidental devolution

        # continue_training = input("Continue? [Y/N]: ")
        # if continue_training.lower() == "n":
        #     break

    # Save the model
    torch.save(winner_model.state_dict(), "/home/maxave/Desktop/connect4-py/models/nn-4")
    return
    '''

    loaded_model = neural_network.NN()
    loaded_model.load_state_dict(torch.load("models/nn-4"))
    loaded_model.eval()

    # loaded_model2 = neural_network.NN()
    # loaded_model2.load_state_dict(torch.load("models/nn-1"))
    # loaded_model2.eval()

    # Player vs AI
    while board.get_winner() == Board.NONE and board.is_draw() == False:
        os.system("clear")
        print(board)

        # Computer move
        print("Computer is thinking...")
        cpu_move = loaded_model.minimax(board, 6, True, float("-inf"), float("inf"))
        print(f"Deep eval: {cpu_move[0]}\nBest move: {cpu_move[1]}")
        board.place(cpu_move[1], Board.X)

        # # Computer move
        # print("Computer is thinking...")
        # cpu_move = loaded_model2.minimax(board, 6, True, float("-inf"), float("inf"))
        # print(f"Deep eval: {cpu_move[0]}\nBest move: {cpu_move[1]}")
        # board.place(cpu_move[1], Board.X)
        
        os.system("clear")
        print(board)

        # Player move
        col = int(input("COL: "))
        col = min(7, max(1, col)) # Only columns 1 - 7
        if board.is_column_free(col - 1):
            board.place(col - 1, Board.O) # Place piece at the selected column
        else:
            print(f"Column {col - 1} is not free. Please choose another one.")
            continue

    # End of game
    print(board)
    if board.is_draw():
        print("IT'S A DRAW!")
    else:
        print("PLAYER 1 (X) WINS!" if board.get_winner() == Board.X else "PLAYER 2 (O) WINS!")


if __name__ == "__main__":
    main()