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
    
class MCTS:

    def __init__(self) -> None:
        self.root: 'TreeNode' = TreeNode(BLACK)
        self.root.set_parent(self.root)
        self.toplay: GO_COLOR = BLACK
    
    def search(self, board: GoBoard, color: GO_COLOR) -> None:
        """
        Run a single playout from the root to the given depth, getting a value at the leaf and
        propagating it back through its parents. State is modified in-place, so a copy must be
        provided.
        Arguments:
        board -- a copy of the board.
        color -- color to play
        """
        node = self.root
        # This will be True only once for the root
        if not node.expanded:
            node.expand(board, color)
        while not node.is_leaf():
            move, next_node = node.select_in_tree(self.exploration)
            assert board.play_move(move, color)
            color = opponent(color)
            node = next_node
        if not node.expanded:
            node.expand(board, color)
        
        assert board.current_player == color
        winner = self.rollout(board, color)
        node.update(winner)
    
    def rollout(self, board: GoBoard, color: GO_COLOR) -> GO_COLOR:
        """
        Use the rollout policy to play until the end of the game, returning the winner of the game
        +1 if black wins, +2 if white wins, 0 if it is a tie.
        """
        # FeatureMoves.playGame will run a simulation game according to given parameters.
        winner = FeatureMoves.playGame(
            board,
            color,
            limit=self.limit, # have no idea what this means
            use_pattern=self.use_pattern,
        )
        return winner
    
    def get_move(
        self,
        board: GoBoard,
        color: GO_COLOR,
        # kinda confused on everything below
        limit: int, 
        use_pattern: bool,
        num_simulation: int,
        exploration: float,
        simulation_policy: str,
        in_tree_knowledge: bool,
    ) -> GO_POINT:
        """
        Runs all playouts sequentially and returns the most visited move.
        """
        if self.toplay != color:
            sys.stderr.write("Tree is for wrong color to play. Deleting.\n")
            sys.stderr.flush()
            self.toplay = color
            self.root = TreeNode(color)
        # confused by what all this means below
        self.limit = limit
        self.use_pattern = use_pattern
        self.exploration = exploration
        self.simulation_policy = simulation_policy
        self.in_tree_knowledge = in_tree_knowledge

        if not self.root.expanded:
            self.root.expand(board, color)

        for _ in range(num_simulation*len(self.root.children)):
            cboard = board.copy()
            self.search(cboard, color)
        # choose a move that has the most visit
        best_move, best_child = self.root.select_best_child()
        return best_move
    
    def update_with_move(self, last_move: GO_POINT) -> None:
        """
        Step forward in the tree, keeping everything we already know about the subtree, assuming
        that get_move() has been called already. Siblings of the new root will be garbage-collected.
        """
        if last_move in self.root.children:
            self.root = self.root.children[last_move]
        else:
            self.root = TreeNode(opponent(self.toplay))
        self.root.parent = self.root
        self.toplay = opponent(self.toplay)

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
