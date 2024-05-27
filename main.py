from connect4 import *
import computer_move
from evaluation import heuristics
from evaluation import neural_network
import threading

def train_with_heuristics(model, score_tally, game_id):
    board = Board()
    computer_move.set_evaluation_function(heuristics.eval)
    moves_played = 0

    try:
        while board.get_winner() == Board.NONE and board.is_draw() == False:
            # Computer move (heuristics)
            best_move = computer_move.minimax(board, 4, True, float("-inf"), float("inf"))
            board.place(best_move[1], Board.X)
            # Computer move (neural network)
            best_move = model.minimax(board, 4, False, float("-inf"), float("inf"))
            board.place(best_move[1], Board.O)

            moves_played += 1
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

    breedable_population_size = 5 # Amount of models that will be allowed to breed after each round
    offspring_population_size = int(breedable_population_size * (breedable_population_size - 1))

    # Create initial population
    active_models = []
    for _ in range(offspring_population_size):
        model = neural_network.NN()
        active_models.append(model)

    for r in range(5):
        print(f"Starting round {r+1}.")

        score_tally = [0] * len(active_models)

        # Let every model play 1 round each with the heuristics bot
        for i in range(len(active_models)):
            train_with_heuristics(active_models[i], score_tally, i)

        # Pick out the best ones
        best_models = []
        while len(best_models) < breedable_population_size:
            max_score = float("-inf")
            best_model_index = 0
            for i in range(len(active_models)):
                if score_tally[i] > max_score:
                    max_score = score_tally[i]
                    best_model_index = i
            best_models.append(active_models[best_model_index])
            del active_models[best_model_index]
            del score_tally[best_model_index]
        active_models.clear()

        # Calculate average
        score_sum = 0
        for score in score_tally:
            score_sum += score
        print(f"Average playing strength: {score_sum / float(len(score_tally))} ({int(score_sum / float(len(score_tally))*100)}%)")
        print(f"Best: {max(score_tally)} ({int(max(score_tally)*100)}%)")
        print(f"Worst: {min(score_tally)} ({int(min(score_tally)*100)}%)")

        # Breed everybody with everybody else
        print("Breeding...")
        for i in range(len(best_models)):
            for j in range(len(best_models)):
                if i != j:
                    active_models.append(neural_network.breed(best_models[i], best_models[j], 0.05))

        continue_training = input("Continue? [Y/N]: ")
        if continue_training.lower() == "n":
            break

    # computer_move.set_evaluation_function(heuristics.eval)
    
    # while board.get_winner() == Board.NONE and board.is_draw() == False:
    #     print(board)
    #     print(f"Shallow eval: {computer_move.evlauation_function(board)}")

    #     # Player move
    #     # col = int(input("COL: "))
    #     # col = min(6, max(0, col)) # Only columns 0 - 6
    #     # if board.is_column_free(col):
    #     #     board.place(col, Board.X) # Place piece at the selected column
    #     # else:
    #     #     print(f"Column {col} is not free. Please choose another one.")
    #     #     continue

    #     computer_move.set_evaluation_function(heuristics.eval)
    #     cpu_move = computer_move.minimax(board, 4, True, float("-inf"), float("inf"))
    #     board.place(cpu_move[1], Board.X)

    #     # Computer move
    #     print("Computer is thinking...")
    #     computer_move.set_evaluation_function(neural_network.eval)
    #     cpu_move = computer_move.minimax(board, 4, False, float("-inf"), float("inf"))
    #     print(f"Deep eval: {cpu_move[0]}; Best move: {cpu_move[1]}")
    #     board.place(cpu_move[1], Board.O)

    # # End of game
    # print(board)
    # if board.is_draw():
    #     print("IT'S A DRAW!")
    # else:
    #     print("PLAYER 1 (X) WINS!" if board.get_winner() == Board.X else "PLAYER 2 (O) WINS!")


if __name__ == "__main__":
    main()