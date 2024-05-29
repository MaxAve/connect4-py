class Board:
    NONE = 0
    X = 1
    O = -1

    def __init__(self):
        self.free_cells = [5, 5, 5, 5, 5, 5, 5]
        # Note: this is a column-major representation (e.g.: [0, 0] = top cell of left-most column)
        self.board = [[0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0]]
    
    def __str__(self):
        board_data = ""
        for i in range(len(self.board[0])):
            for column in self.board:
                if column[i] == Board.X:
                    board_data += "X "
                elif column[i] == Board.O:
                    board_data += "\033[93mO\033[0m "
                else:
                    board_data += "- "
            board_data += "\n"
        return f"1 2 3 4 5 6 7\n{board_data}"
    
    def is_column_free(self, column):
        return self.free_cells[column] > -1
    
    def get_free_columns(self):
        fcols = []
        for col in range(7):
            if self.is_column_free(col):
                fcols.append(col)
        return fcols

    def place(self, column, piece):
        if not self.is_column_free(column):
            self.place(self.get_free_columns[0])
            return
        self.board[column][self.free_cells[column]] = piece
        self.free_cells[column] -= 1
    
    def get_winner(self):
        # |
        for col in range(7):
            for row in range(3):
                if self.board[col][row] != Board.NONE:
                    if self.board[col][row] == self.board[col][row + 1] == self.board[col][row + 2] == self.board[col][row + 3]:
                        return self.board[col][row]
        # -
        for row in range(6):
            for col in range(4):
                if self.board[col][row] != Board.NONE:
                    if self.board[col][row] == self.board[col + 1][row] == self.board[col + 2][row] == self.board[col + 3][row]:
                        return self.board[col][row]
        # /
        for row in range(3, 6):
            for col in range(4):
                if self.board[col][row] != Board.NONE:
                    if self.board[col][row] == self.board[col + 1][row - 1] == self.board[col + 2][row - 2] == self.board[col + 3][row - 3]:
                        return self.board[col][row]
        # \
        for row in range(3):
            for col in range(4):
                if self.board[col][row] != Board.NONE:
                    if self.board[col][row] == self.board[col + 1][row + 1] == self.board[col + 2][row + 2] == self.board[col + 3][row + 3]:
                        return self.board[col][row]
        return Board.NONE
    
    def is_draw(self):
        return sum(self.free_cells) == -6