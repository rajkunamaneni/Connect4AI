import numpy as np


class AIPlayer:
    def __init__(self, player_number):
        self.player_number = player_number
        self.type = 'ai'
        self.depth = 0
        self.maxdepth = 4
        self.mindepth = 0
        self.player_string = 'Player {}:ai'.format(player_number)

    def get_alpha_beta_move(self, board):
        """
        Given the current state of the board, return the next move based on
        the alpha-beta pruning algorithm

        This will play against either itself or a human player

        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The 0 based index of the column that represents the next move
        """
        def find_move(board):
            valid_cols = []
            for col in range(board.shape[1]):
                for row in range(board.shape[0]):
                    if board[row][col] == 0:
                        valid_cols.append([row, col])
            return valid_cols

        def minvalue(board, alpha, beta, player, oppenent):
            x = np.inf
            if self.depth == self.maxdepth or not find_move(board):
                return self.evaluation_function(board)
            for move in find_move(board):
                r, c = move
                board[r][c] = oppenent
                self.depth += 1
                x = min(x, maxvalue(board, alpha, beta, player, oppenent))
                board[r][c] = 0
                if x <= alpha:
                    return x
            return x

        def maxvalue(board, alpha, beta, player, oppenent):
            x = -np.inf
            if self.depth == self.maxdepth or find_move(board):
                return self.evaluation_function(board)
            for move in find_move(board):
                r, c = move
                board[r][c] = player
                self.depth += 1
                x = max(x, minvalue(board, alpha, beta, player, oppenent))
                board[r][c] = 0
                if x >= beta:
                    return x
            return x

        alpha, beta, vltype, player = -np.inf, np.inf, list(), self.player_number
        opponent = 2 if player == 1 else 1
        for move in find_move(board):
            r, c = move
            board[r][c] = player
            self.depth += 1

            alpha = max(alpha, minvalue(board, alpha, beta, player, opponent))
            vltype.append((alpha, c))
            board[r][c] = 0
        value = (max(vltype, key=lambda x: x[1])[0])
        for type in vltype:
            if value in type:
                return (type[1])

    def get_expectimax_move(self, board):
        """
        Given the current state of the board, return the next move based on
        the expectimax algorithm.

        This will play against the random player, who chooses any valid move
        with equal probability

        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The 0 based index of the column that represents the next move
        """

        def find_move(board):
            valid_cols = []
            for col in range(board.shape[1]):
                for row in range(board.shape[0]):
                    if board[row][col] == 0:
                        valid_cols.append([row, col])
                        break
            return valid_cols

        def maxvalue(board, player, oppenent, depth):
            x = -np.inf
            if depth == self.mindepth or not find_move(board):
                return (self.evaluation_function(board))
            for move in find_move(board):
                r, c = move
                board[r][c] = player
                x = max(x, expectedval(board, player, oppenent, depth-1))
            return x

        def expectedval(board, player, oppenent, depth):
            x, savep = 0, len(find_move(board))
            if depth == self.mindepth or not find_move(board):
                return (self.evaluation_function(board))
            for move in find_move(board):
                r, c = move
                board[r][c] = oppenent
                x += maxvalue(board, player, oppenent, depth-1)
            return (x/savep)

        depth, vltype, player, x = 8, list(), self.player_number, -np.inf
        opponent = 2 if player == 1 else 1
        for move in find_move(board):
            r, c = move
            board[r][c] = player
            x = max(x, expectedval(board, player, opponent, depth-1))
            vltype.append((x, c))
            board[r][c] = 0
        value = (max(vltype, key=lambda x: x[1])[0])
        for type in vltype:
            if value in type:
                return (type[1])

    def evaluation_function(self, board):
        """
        Given the current stat of the board, return the scalar value that 
        represents the evaluation function for the current player

        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The utility value for the current board
        """

        def somecounter(board, val, pnum):
            winstrategy, to_str = '{0}' * val, lambda a: ''.join(a.astype(str))
            winstrategy = winstrategy.format(pnum)

            def check_horizontal(b):
                contreturn = 0
                for row in b:
                    if winstrategy in to_str(row):
                        contreturn += to_str(row).count(winstrategy)
                return contreturn

            def check_verticle(b):
                return check_horizontal(b.T)

            def check_diagonal(b):
                contreturn = 0
                for op in [None, np.fliplr]:
                    op_board = op(b) if op else b
                    root_diag = np.diagonal(op_board, offset=0).astype(np.int)
                    if winstrategy in to_str(root_diag):
                        contreturn += to_str(root_diag).count(winstrategy)

                    for i in range(1, b.shape[1]-3):
                        for offset in [i, -i]:
                            diag = np.diagonal(op_board, offset=offset)
                            diag = to_str(diag.astype(np.int))
                            if winstrategy in diag:
                                contreturn += diag.count(winstrategy)
                return contreturn

            return check_horizontal(board) + check_verticle(board) + check_diagonal(board)

        player, result, value = self.player_number, 0, 9050
        opponent = 2 if player == 1 else 1
        for i in range(4, 1, -1):
            result += somecounter(board, i, player) * value
            value /= 10  # devalue the cost

        value = 9000
        for i in range(4, 1, -1):
            result -= somecounter(board, i, opponent) * value
            value /= 10
        return (result)  # devalue the cost


class RandomPlayer:
    def __init__(self, player_number):
        self.player_number = player_number
        self.type = 'random'
        self.player_string = 'Player {}:random'.format(player_number)

    # random player input
    def get_move(self, board):
        """
        Given the current board state select a random column from the available
        valid moves.

        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The 0 based index of the column that represents the next move
        """
        valid_cols = []
        for col in range(board.shape[1]):
            if 0 in board[:, col]:
                valid_cols.append(col)
        return np.random.choice(valid_cols)


class HumanPlayer:
    def __init__(self, player_number):
        self.player_number = player_number
        self.type = 'human'
        self.player_string = 'Player {}:human'.format(player_number)

    # human player input
    def get_move(self, board):
        """
        Given the current board state returns the human input for next move

        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The 0 based index of the column that represents the next move
        """
        valid_cols = []
        for i, col in enumerate(board.T):
            if 0 in col:
                valid_cols.append(i)

        move = int(input('Enter your move: '))

        while move not in valid_cols:
            print('Column full, choose from:{}'.format(valid_cols))
            move = int(input('Enter your move: '))
        return move
