#!/usr/bin/python3
# Set the path to your python3 above

"""
Cmput 455 sample code
Written by Cmput 455 TA and Martin Mueller
"""
from gtp_connection import GtpConnection, format_point, point_to_coord
from board_base import DEFAULT_SIZE, GO_POINT, GO_COLOR
from board import GoBoard
from board_util import GoBoardUtil
from engine import GoEngine
import copy

from board_base import (
    BLACK,
    WHITE,
    EMPTY,
    BORDER,
    GO_COLOR, GO_POINT,
    PASS,
    MAXSIZE,
    coord_to_point,
    opponent
)


class SimulationPlayer(GoEngine):
    def __init__(self, numSimulations):
        self.numSimulations = numSimulations

    def name(self):
        return "Simulation Player ({0} sim.)".format(self.numSimulations)

    def genmove(self, state: GoBoard):
        moves = state.get_empty_points()
        numMoves = len(moves)
        score = [0] * numMoves
        for i in range(numMoves):
            move = moves[i]
            score[i] = self.simulate(state, move)
        bestIndex = score.index(max(score))
        best = moves[bestIndex]
        return best

    def simulate(self, state: GoBoard, move: GO_POINT):
        stats = [0] * 3
        state_copy = copy.deepcopy(state)
        state_copy.play_move(move, state.current_player)
        for i in range(self.numSimulations):
            winner = state_copy.simulateMoves(move)
            stats[winner] += 1
            state_copy = copy.deepcopy(state)
        
        assert sum(stats) == self.numSimulations
        eval = (stats[BLACK] + 0.5 * stats[EMPTY]) / self.numSimulations
        if state.current_player == WHITE:
            eval = 1 - eval
        return eval

def run() -> None:
    """
    start the gtp connection and wait for commands.
    """
    board: GoBoard = GoBoard(DEFAULT_SIZE)
    sim: SimulationPlayer = SimulationPlayer(10)
    con: GtpConnection = GtpConnection(sim, board)
    con.start_connection()

if __name__ == "__main__":
    run()
