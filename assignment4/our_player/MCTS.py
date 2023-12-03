import numpy as np
import math
from board import GoBoard
import copy
import gtp_connection
import time
#from utils import print_pi
EPS = 1e-8

from gtp_connection import GtpConnection

class MCTS:

    def __init__(self, board: GoBoard):
        #self.game = game    # NogoGame instance
        #self.nnet = nnet    # NNetWrapper instance
        #self.args = args
        self.cuct = 1
        self.board: GoBoard = board
        self.time_start = 0
        self.max_time = 55

        # s ==> state or game position
        # a ==> action or legal move
        # Qsa ==> value of playing a move a in game position s, is the win rate
        # Nsa ==> number of times a move has been played 
        # Ns ==> number of times a game position has been visited
        # Ps ==> what the neural networl thinks the policy should be --> NOT NEEDED
        # Vs ==> valid moves at game position s
        # Es ==> whether game position s i s a terminal state

        self.Qsa = {}       # store Q values for s,a (as defined in the paper)
        self.Nsa = {}       # stores #times edge s,a was visited
        self.Ns = {}        # stores #times board s was visited
        #self.Ps = {}        # stores initial policy (returned by neural net)
        self.Vs = {}        # stores game.getValidMoves for board s
        self.Es = {}        # stores game.getGameEnded ended for board s

    def getActionProb(self, board_copy):
        """
        This function performs numMCTSSims simulations starting from
        canonicalBoard.

        Inputs:
            canonicalBoard: the canonical form of the board

        Returns:
            probs: a policy vector
        """
        print("############################## /n /n ")
        #self.game.beginSearch()
        #for _ in range(self.args.numMCTSSims): # change to a while loop where the end condition is our time limit
        while ((time.process_time()-self.time_start) >= self.max_time):
            #self.game.inSearch()
            self.search(self.board)
        #self.game.endSearch()

        #s = self.game.stringRepresentation(canonicalBoard)
        board_copy = copy.deepcopy(self.board)
        s = board_copy
        counts = [self.Nsa[(s,a)] if (s,a) in self.Nsa else 0 for a in range(len(self.board.get_empty_points()))]
        counts_sum = float(np.sum(counts))
        if counts_sum == 0.0:
            return None
        probs = [x/counts_sum for x in counts]

        return probs
    
    def search(self, board_copy):
        """
        This function performs one simulation.

        Inputs:
            canonicalBoard: the canonical form of the board (not used)

        Returns:
            -v: the negative value of the win/loss w.r.t. the current player
        """
        #canonicalBoard = self.game.getCanonicalBoard(canonicalBoard, 1)  # inputs ignored --> ignore getCanonicalBoard(), its for neural network so we dont have to worry about using this
        board_copy = copy.deepcopy(self.board)
        s = board_copy
        if s in self.Es: # checks if the state is a terminal node
            # terminal node
            return -self.Es[s]

        #if s not in self.Ps: # if state is not apart of a policy ??? MORE CLARIFICATION
        if GtpConnection.policy_moves() == False:
            # leaf node
            valids = self.board.get_empty_points() # valid moves --> legal moves / all empty moves
            self.Vs[s] = valids
            self.Ns[s] = 0 # sets the number of times this state has been visited to 0

            if self.Vs[s].max() == 0:   # terminal state --> just import python library
                self.Es[s] = -1
                return -self.Es[s]

            
            """self.Ps[s] = self.Ps[s] * valids    # masking invalid moves, change to array, instead of a vector
                                                # this basically fills illegal moves with 0 so our simulation will never play at illegal moves
            sum_Ps_s = np.sum(self.Ps[s])
            if sum_Ps_s > 0:
                self.Ps[s] /= sum_Ps_s      # re-normalize
            else:
                print("All valid moves were masked by NN")
                self.Ps[s] = self.Ps[s] + valids
                self.Ps[s] /= np.sum(self.Ps[s])"""

            return -v

        best_a = -1
        best_u = -float("inf")
        # pick the action with the highest upper confidence bound
        # set p (probility vairable in PUCT) to 1 --> p == self.Ps[s][a]
        for a in range(len(self.board.get_empty_points())): # getActionSize() just returns a number for all possible actions
            # for move in range of all empty points
            if self.Vs[s][a]:
                # if move is a valid moves at game position s
                if (s,a) in self.Qsa:
                    # if state and move has a win rate
                    # self.Ps[s][a] = 1 # set to 1 so it does not affect the uct formula
                    # u = self.Qsa[(s,a)] + self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (1+self.Nsa[(s,a)]) # UCT formula to get upper confidence bound
                    u = self.Qsa[(s,a)] + self.cuct * math.sqrt(self.Ns[s]) / (1+self.Nsa[(s,a)]) # UCT formula to get upper confidence bound
                else:
                    # if hasnt been visited yet???
                    # u = 1.0 * self.Ps[s][a] * math.sqrt(self.Ns[s]+EPS)     # Q = 0 ?
                    u = 1.0 * math.sqrt(self.Ns[s]+EPS)     # Q = 0 ?

                if u > best_u:
                    # if uct is better then a uct we have so far will update for best confidence and best move
                    best_u = u
                    best_a = a
        
        a = best_a
        assert a != -1

        # next_s, next_player = self.game.getNextState(canonicalBoard, 1, a) # gets the next state after opponent plays
        
        next_s = self.board.play_move(1, a)

        v = self.search(next_s) # recursive --> finds value of win/loss for current player

        if (s,a) in self.Qsa:
            # if move and state has a winrate then will update winrate and number of times a move has been played
            self.Qsa[(s,a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s,a)] += 1
        else:
            self.Qsa[(s,a)] = v
            self.Nsa[(s,a)] = 1

        self.Ns[s] += 1
        return -v
    
    def clear(self):
        self.Qsa.clear()
        self.Nsa.clear()
        self.Ns.clear()
        #self.Ps.clear()
        self.Vs.clear()
        self.Es.clear()
        return
