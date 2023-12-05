"""
board.py
Cmput 455 sample code
Written by Cmput 455 TA and Martin Mueller

Implements a basic Go board with functions to:
- initialize to a given board size
- check if a move is legal
- play a move

The board uses a 1-dimensional representation with padding
"""

import numpy as np
from typing import List, Tuple
import random

from board_base import (
    board_array_size,
    coord_to_point,
    is_black_white,
    is_black_white_empty,
    opponent,
    where1d,
    BLACK,
    WHITE,
    EMPTY,
    BORDER,
    MAXSIZE,
    NO_POINT,
    PASS,
    GO_COLOR,
    GO_POINT,
)


"""
The GoBoard class implements a board and basic functions to play
moves, check the end of the game, and count the acore at the end.
The class also contains basic utility functions for writing a Go player.
For many more utility functions, see the GoBoardUtil class in board_util.py.

The board is stored as a one-dimensional array of GO_POINT in self.board.
See coord_to_point for explanations of the array encoding.
"""
class GoBoard(object):
    def __init__(self, size: int) -> None:
        """
        Creates a Go board of given size
        """
        assert 2 <= size <= MAXSIZE
        self.reset(size)
        self.black_captures = 0
        self.white_captures = 0
        self.depth = 0
        self.black_capture_history = []
        self.white_capture_history = []
        self.move_history = []

    def add_two_captures(self, color: GO_COLOR) -> None:
        if color == BLACK:
            self.black_captures += 2
        elif color == WHITE:
            self.white_captures += 2
    
    def get_captures(self, color: GO_COLOR) -> None:
        if color == BLACK:
            return self.black_captures
        elif color == WHITE:
            return self.white_captures
    
    def reset(self, size: int) -> None:
        """
        Creates a start state, an empty board with given size.
        """
        self.size: int = size
        self.NS: int = size + 1
        self.WE: int = 1
        self.last_move: GO_POINT = NO_POINT
        self.last2_move: GO_POINT = NO_POINT
        self.current_player: GO_COLOR = BLACK
        self.maxpoint: int = board_array_size(size)
        self.board: np.ndarray[GO_POINT] = np.full(self.maxpoint, BORDER, dtype=GO_POINT)
        self._initialize_empty_points(self.board)
        self.black_captures = 0
        self.white_captures = 0
        self.depth = 0
        self.black_capture_history = []
        self.white_capture_history = []
        self.move_history = []

    def copy(self) -> 'GoBoard':
        b = GoBoard(self.size)
        assert b.NS == self.NS
        assert b.WE == self.WE
        b.last_move = self.last_move
        b.last2_move = self.last2_move
        b.current_player = self.current_player
        assert b.maxpoint == self.maxpoint
        b.board = np.copy(self.board)
        b.black_captures = self.black_captures
        b.white_captures = self.white_captures
        b.depth = self.depth
        b.black_capture_history = self.black_capture_history.copy()
        b.white_capture_history = self.white_capture_history.copy()
        b.move_history = self.move_history.copy()
        return b

    def get_color(self, point: GO_POINT) -> GO_COLOR:
        return self.board[point]

    def pt(self, row: int, col: int) -> GO_POINT:
        return coord_to_point(row, col, self.size)

    def is_legal(self, point: GO_POINT, color: GO_COLOR) -> bool:
        """
        Check whether it is legal for color to play on point
        This method tries to play the move on a temporary copy of the board.
        This prevents the board from being modified by the move
        """
        if point == PASS:
            return True
        #board_copy: GoBoard = self.copy()
        #can_play_move = board_copy.play_move(point, color)
        #return can_play_move
        return self.board[point] == EMPTY

    def end_of_game(self) -> bool:
        return self.get_empty_points().size == 0 or (self.last_move == PASS and self.last2_move == PASS)
           
    def get_empty_points(self) -> np.ndarray:
        """
        Return:
            The empty points on the board
        """
        return where1d(self.board == EMPTY)

    def check_board_empty(self) -> bool:
        return np.all(self.board == EMPTY) 

    def check_corner_empty(self):
        corners = [(1,1),(1,self.size), (self.size,1),(self.size, self.size)]
        for c in corners:
            row, col = c
            if self.get_color(self.pt(row,col)) != EMPTY:
                return True
    
    def check_centre_empty(self):
        centre = (self.size // 2+1, self.size // 2+1)
        return self.get_color(self.pt(*centre)) == EMPTY

    def centre_play(self):
        centre = (self.size // 2+1, self.size // 2+1)
        self.play_move(self.pt(*centre), self.current_player)

    def corner_play(self):
        empty_corner = [(1,1),(1,self.size), (self.size,1),(self.size, self.size)]
        random.shuffle(empty_corner)
        for c in empty_corner:
            row, col = c
            if self.get_color(self.pt(row,col)) == EMPTY:
                self.play_move(self.pt(row,col), self.current_player)
                break
    
    def play_next_to_last_player_move(self):
        if self.last_move != NO_POINT and self.last_move != PASS:
            neighbors = self._neighbors(self.last_move)
            empty_neighbors = [nb for nb in neighbors if self.get_color(nb)]
            if empty_neighbors == EMPTY:
                random.shuffle(empty_neighbors)
                self.play_move(empty_neighbors[0], self.current_player)

    def row_start(self, row: int) -> int:
        assert row >= 1
        assert row <= self.size
        return row * self.NS + 1

    def _initialize_empty_points(self, board_array: np.ndarray) -> None:
        """
        Fills points on the board with EMPTY
        Argument
        ---------
        board: numpy array, filled with BORDER
        """
        for row in range(1, self.size + 1):
            start: int = self.row_start(row)
            board_array[start : start + self.size] = EMPTY

    def play_move(self, point: GO_POINT, color: GO_COLOR) -> bool:
        """
        Tries to play a move of color on the point.
        Returns whether or not the point was empty.
        """
        if self.board[point] != EMPTY:
            return False
        self.board[point] = color
        self.current_player = opponent(color)
        self.last2_move = self.last_move
        self.last_move = point
        O = opponent(color)
        offsets = [1, -1, self.NS, -self.NS, self.NS+1, -(self.NS+1), self.NS-1, -self.NS+1]
        bcs = []
        wcs = []
        for offset in offsets:
            if self.board[point+offset] == O and self.board[point+(offset*2)] == O and self.board[point+(offset*3)] == color:
                self.board[point+offset] = EMPTY
                self.board[point+(offset*2)] = EMPTY
                if color == BLACK:
                    self.black_captures += 2
                    bcs.append(point+offset)
                    bcs.append(point+(offset*2))
                else:
                    self.white_captures += 2
                    wcs.append(point+offset)
                    wcs.append(point+(offset*2))
        self.depth += 1
        self.black_capture_history.append(bcs)
        self.white_capture_history.append(wcs)
        self.move_history.append(point)
        return True
    
    def undo(self):
        self.board[self.move_history.pop()] = EMPTY
        self.current_player = opponent(self.current_player)
        self.depth -= 1
        bcs = self.black_capture_history.pop()
        for point in bcs:
            self.board[point] = WHITE
            self.black_captures -= 1
        wcs = self.white_capture_history.pop()
        for point in wcs:
            self.board[point] = BLACK
            self.white_captures -= 1
        if len(self.move_history) > 0:
            self.last_move = self.move_history[-1]
        if len(self.move_history) > 1:
            self.last2_move = self.move_history[-2]

    def neighbors_of_color(self, point: GO_POINT, color: GO_COLOR) -> List:
        """ List of neighbors of point of given color """
        nbc: List[GO_POINT] = []
        for nb in self._neighbors(point):
            if self.get_color(nb) == color:
                nbc.append(nb)
        return nbc

    def _neighbors(self, point: GO_POINT) -> List:
        """ List of all four neighbors of the point """
        return [point - 1, point + 1, point - self.NS, point + self.NS]

    def _diag_neighbors(self, point: GO_POINT) -> List:
        """ List of all four diagonal neighbors of point """
        return [point - self.NS - 1,
                point - self.NS + 1,
                point + self.NS - 1,
                point + self.NS + 1]

    def last_board_moves(self) -> List:
        """
        Get the list of last_move and second last move.
        Only include moves on the board (not NO_POINT, not PASS).
        """
        board_moves: List[GO_POINT] = []
        if self.last_move != NO_POINT and self.last_move != PASS:
            board_moves.append(self.last_move)
        if self.last2_move != NO_POINT and self.last2_move != PASS:
            board_moves.append(self.last2_move)
        return board_moves

    def play_move_conditions(self):
        if np.all(self.board == EMPTY):
        #if board empty then play in the centre
            centre = (self.size // 2 + 1, self.size // 2 + 1)
            self.play_move(self.pt(*centre), self.current_player)
        elif all(self.get_color(self.pt(row, col)) != EMPTY for row, col in [(1, 1), (1, self.size), (self.size, 1), (self.size, self.size)]):
        # If all four corners are full, play in the centre
            centre = (self.size // 2 + 1, self.size // 2 + 1)
            self.play_move(self.pt(*centre), self.current_player)
        elif self.get_color(self.pt(self.size // 2 + 1, self.size // 2 + 1)) != EMPTY:
        # If centre is full, play in the empty corner
            empty_corners = [(1, 1), (1, self.size), (self.size, 1), (self.size, self.size)]
            random.shuffle(empty_corners)
            for c in empty_corners:
                row, col = c
                if self.get_color(self.pt(row, col)) == EMPTY:
                    self.play_move(self.pt(row, col), self.current_player)
                    return
        else:
        # If both corners and the centre are full, play next to the last player's move
            if self.last_move != NO_POINT and self.last_move != PASS:
                neighbors = self._neighbors(self.last_move)
                empty_neighbors = [nb for nb in neighbors if self.get_color(nb) == EMPTY]
                if empty_neighbors:
                    random.shuffle(empty_neighbors)
                    self.play_move(empty_neighbors[0], self.current_player)


    def Win(self):
        winning_moves = []
        legal_moves = self.get_empty_points()
        player_color = self.current_player

        for move in legal_moves:    
            # check for 5 in a row
            self.play_move(move, player_color)
            color = self.detect_five_in_a_row()
            if color == player_color:
                winning_moves.append(move)

            # check captures
            if player_color == WHITE and self.white_captures >= 10:
                    winning_moves.append(move)
                    self.white_captures -= 2
            if player_color == BLACK and self.black_captures >= 10:
                    winning_moves.append(move)
                    self.black_captures -= 2

            # reset points filled back to empty
            self.board[move] = EMPTY

        return winning_moves
        
    def BlockWin(self):
        winPoints = []
        empty_points = self.get_empty_points()

        # check if there is a five in a row
        for point in empty_points:
            if self.current_player == WHITE:
                previous_captures = self.black_captures
            else:
                previous_captures = self.white_captures
            
            self.play_move(point, opponent(self.current_player))
            color = self.detect_five_in_a_row()

            if color == opponent(self.current_player):
                winPoints.append(point)
            
            if previous_captures == 8:
                if self.current_player == WHITE and previous_captures < self.black_captures:
                    winPoints.append(point)
                    self.black_captures -= 2
                elif self.current_player == BLACK and previous_captures < self.white_captures:
                    winPoints.append(point)
                    self.white_captures -= 2
            self.board[point] = EMPTY

        return winPoints
    
    def OpenThree(self):
        open_three_moves = []
        legal_moves = self.get_empty_points()
        player_color = self.current_player
        # check in each row, col, and diagonal if there is an open three
        for move in legal_moves:
            board_copy = self.board.copy()
            self.play_move(move, player_color)
            color = self.detect_three_in_a_row(move)
            if color == player_color:
                open_three_moves.append(move)
            self.board[move] = EMPTY
            self.board = board_copy

        return open_three_moves

    def DoubleOpenThree(self):
        double_open_three_moves = []
        legal_moves = self.get_empty_points()
        player_color = self.current_player
        for move in legal_moves:
            board_copy = self.board.copy()
            self.play_move(move, player_color)
            color = self.detect_three_in_a_row(move)
        
            if color == player_color:
                for second_move in legal_moves:
                    if second_move != move:
                        self.play_move(second_move, player_color)
                        second_color = self.detect_three_in_a_row(second_move)
                        if second_color == player_color:
                            double_open_three_moves.append((move, second_move))
                        self.board[second_move] = EMPTY
            self.board[move] = EMPTY
            self.board = board_copy
    
        return double_open_three_moves

    def OpenFour(self):
        open_four_moves = []
        legal_moves = self.get_empty_points()
        player_color = self.current_player
        # check in each row, col, and diagonal if there is 4 in a list
        for move in legal_moves:
            board_copy = self.board.copy()
            self.play_move(move, player_color)
            color = self.detect_four_in_a_row(move)
            if color == player_color:
                open_four_moves.append(move)
            self.board[move] = EMPTY
            self.board = board_copy

        return open_four_moves

    def DoubleOpenFour(self):
        double_open_four_moves = []
        legal_moves = self.get_empty_points()
        player_color = self.current_player
        for move in legal_moves:
            board_copy = self.board.copy()
            self.play_move(move, player_color)
            color = self.detect_four_in_a_row(move)
        
            if color == player_color:
                for second_move in legal_moves:
                    if second_move != move:
                        self.play_move(second_move, player_color)
                        second_color = self.detect_four_in_a_row(second_move)
                        if second_color == player_color:
                            double_open_four_moves.append((move, second_move))
                        self.board[second_move] = EMPTY
            self.board[move] = EMPTY
            self.board = board_copy
    
        return double_open_four_moves   
        
    def Capture(self):
        captured_moves = []
        legal_moves = self.get_empty_points()
        player_color = self.current_player#doubt = when we were playing it. it was alternating players with only self.current player, so shoudld it do that or it should be consistent??

        for move in legal_moves:#to play a move
            
            if player_color == WHITE:#previous_captures is there so we can chcek stuff later on depending on the player 
                previous_captures = self.white_captures
            else:
                previous_captures = self.black_captures
        
            board_copy = self.board.copy()#a copy of board is created before playing each move. move is played on the copy of the board and original board is restored after checking for captures.
            self.play_move(move,player_color)

            if player_color == WHITE and previous_captures < self.white_captures:#we played a move and white captures would have more than previous captures if it captured something
            #if we did capture in the move we played white should be more  
                captured_moves.append(move)
                self.white_captures -= 2 #resetting the white captures to its original state 
            if player_color == BLACK and previous_captures < self.black_captures:
                captured_moves.append(move)
                self.white_captures -= 2

            self.board = board_copy 
        return captured_moves
            
            
    def Random(self):
        legal_moves = self.get_empty_points()
        return legal_moves

    def detect_three_in_a_row(self, move) -> GO_COLOR:
    
        for r in self.rows:
            result = self.has_three_in_list(r, move)
            if result != EMPTY:
                return result
        for c in self.cols:
            result = self.has_three_in_list(c, move)
            if result != EMPTY:
                return result
        for d in self.diags:
            result = self.has_three_in_list(d, move)
            if result != EMPTY:
                return result
        return EMPTY

    def has_three_in_list(self, list, move) -> GO_COLOR:
        prev = BORDER
        counter = 1
        three_in_list = []
        for stone in list:
            if self.get_color(stone) == prev and self.get_color(stone)!= EMPTY:
                three_in_list.append(stone)
                if len(three_in_list) == 3:
                    if self.get_color(stone-2) == self.get_color(stone):
                        three_in_list.append(stone-2)
                counter += 1
            else:
                counter = 1
                prev = self.get_color(stone)
            if counter == 3 and prev != EMPTY and move in three_in_list:
                return prev
        return EMPTY

    def detect_four_in_a_row(self, move) -> GO_COLOR:
    
        for r in self.rows:
            result = self.has_four_in_list(r, move)
            if result != EMPTY:
                return result
        for c in self.cols:
            result = self.has_four_in_list(c, move)
            if result != EMPTY:
                return result
        for d in self.diags:
            result = self.has_four_in_list(d, move)
            if result != EMPTY:
                return result
        return EMPTY

    def has_four_in_list(self, list, move) -> GO_COLOR:
        prev = BORDER
        counter = 1
        four_in_list = []
        for stone in list:
            if self.get_color(stone) == prev and self.get_color(stone)!= EMPTY:
                four_in_list.append(stone)
                if len(four_in_list) == 3:
                    if self.get_color(stone-3) == self.get_color(stone):
                        four_in_list.append(stone-3)
                counter += 1
            else:
                counter = 1
                prev = self.get_color(stone)
            if counter == 4 and prev != EMPTY and move in four_in_list:
                return prev
        return EMPTY

    def full_board_detect_five_in_a_row(self) -> GO_COLOR:
        """
        Returns BLACK or WHITE if any five in a row is detected for the color
        EMPTY otherwise.
        Checks the entire board.
        """
        for point in range(self.maxpoint):
            c = self.board[point]
            if c != BLACK and c != WHITE:
                continue
            for offset in [(1, 0), (0, 1), (1, 1), (1, -1)]: 
                i = 1
                num_found = 1
                while self.board[point + i * offset[0] * self.NS + i * offset[1]] == c:
                    i += 1
                    num_found += 1
                i = -1
                while self.board[point + i * offset[0] * self.NS + i * offset[1]] == c:
                    i -= 1
                    num_found += 1
                if num_found >= 5:
                    return c
        
        return EMPTY
    
    def detect_five_in_a_row(self) -> GO_COLOR:
        """
        Returns BLACK or WHITE if any five in a row is detected for the color
        EMPTY otherwise.
        Only checks around the last move for efficiency.
        """
        if self.last_move == NO_POINT or self.last_move == PASS:
            return EMPTY
        c = self.board[self.last_move]
        for offset in [(1, 0), (0, 1), (1, 1), (1, -1)]:
            i = 1
            num_found = 1
            while self.board[self.last_move + i * offset[0] * self.NS + i * offset[1]] == c:
                i += 1
                num_found += 1
            i = -1
            while self.board[self.last_move + i * offset[0] * self.NS + i * offset[1]] == c:
                i -= 1
                num_found += 1
            if num_found >= 5:
                return c
        
        return EMPTY

    def has_five_in_list(self, list) -> GO_COLOR:
        """
        Returns BLACK or WHITE if any five in a rows exist in the list.
        EMPTY otherwise.
        """
        prev = BORDER
        counter = 1
        for stone in list:
            if self.get_color(stone) == prev:
                counter += 1
            else:
                counter = 1
                prev = self.get_color(stone)
            if counter == 5  and prev != EMPTY:
                return prev
        return EMPTY

    def is_terminal(self):
        """
        Returns: is_terminal, winner
        If the result is a draw, winner = EMPTY
        """
        winner = self.detect_five_in_a_row()
        if winner != EMPTY:
            return True, winner
        elif self.get_captures(BLACK) >= 10:
            return True, BLACK
        elif self.get_captures(WHITE) >= 10:
            return True, WHITE
        elif self.end_of_game():
            return True, EMPTY
        else:
            return False, EMPTY

    def heuristic_eval(self):
        """
        Returns: a very basic heuristic value of the board
        Currently only considers captures
        """
        if self.current_player == BLACK:
            return (self.black_captures - self.white_captures) / 10
        else:
            return (self.white_captures - self.black_captures) / 10

    def state_to_str(self):
        state = np.array2string(self.board, separator='')
        state += str(self.current_player)
        state += str(self.black_captures)
        state += str(self.white_captures)
        return state
    
    def simulateMoves(self): 
        allMoves = self.get_empty_points()

        #current implementation is random, will need to make the different kinds

        random.shuffle(allMoves)
        i = 0
        while not self.end_of_game():
            self.play_move(allMoves[i],opponent(self.current_player))
            i+= 1
            
        result1 = self.detect_five_in_a_row()

        if self.get_captures(BLACK) >= 10 or result1 == BLACK:
            return BLACK
        elif self.get_captures(WHITE) >= 10 or result1 == WHITE:
            return WHITE
        elif self.get_empty_points().size == 0:
            return EMPTY 
