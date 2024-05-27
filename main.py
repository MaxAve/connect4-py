from connect4 import *
import computer_move
from evaluation import heuristics
import os

def main():
    board = Board()
    current_piece = Board.X
    computer_move.set_evaluation_function(heuristics.eval)

    while board.get_winner() == Board.NONE and board.is_draw() == False:
        print(board)
        print(f"Shallow eval: {heuristics.eval(board)}")

        # Player move
        col = int(input("COL: "))
        col = min(6, max(0, col)) # Only columns 0 - 6
        if board.is_column_free(col):
            board.place(col, Board.X) # Place piece at the selected column
        else:
            print(f"Column {col} is not free. Please choose another one.")
            continue

        # Computer move
        print("Computer is thinking...")
        cpu_move = computer_move.minimax(board, 4, False, float("-inf"), float("inf"))
        print(f"Deep eval: {cpu_move[0]}; Best move: {cpu_move[1]}")
        board.place(cpu_move[1], Board.O)

    # End of game
    print(board)
    if board.is_draw():
        print("IT'S A DRAW!")
    else:
        print("PLAYER 1 (X) WINS!" if board.get_winner() == Board.X else "PLAYER 2 (O) WINS!")


if __name__ == "__main__":
    main()