import numpy as np
import math
#from utils import print_pi
EPS = 1e-8

class MCTS:

    def __init__(self, game, nnet, args):
        self.game = game    # NogoGame instance
        self.nnet = nnet    # NNetWrapper instance
        self.args = args

        self.Qsa = {}       # store Q values for s,a (as defined in the paper)
        self.Nsa = {}       # stores #times edge s,a was visited
        self.Ns = {}        # stores #times board s was visited
        self.Ps = {}        # stores initial policy (returned by neural net)
        self.Vs = {}        # stores game.getValidMoves for board s
        self.Es = {}        # stores game.getGameEnded ended for board s

    def getActionProb(self, canonicalBoard, temp=1, verbose=False):
        """
        This function performs numMCTSSims simulations starting from
        canonicalBoard.

        Inputs:
            canonicalBoard: the canonical form of the board

        Returns:
            probs: a policy vector
        """
        self.game.beginSearch()
        for _ in range(self.args.numMCTSSims): # change to a while loop where the end condition is our time limit
            self.game.inSearch()
            self.search(canonicalBoard)
        self.game.endSearch()

        s = self.game.stringRepresentation(canonicalBoard)
        counts = [self.Nsa[(s,a)] if (s,a) in self.Nsa else 0 for a in range(self.game.getActionSize())]
        counts_sum = float(np.sum(counts))
        if counts_sum == 0.0:
            return None
        probs = [x/counts_sum for x in counts]

        return probs
    
    def search(self, canonicalBoard):
        """
        This function performs one simulation.

        Inputs:
            canonicalBoard: the canonical form of the board (not used)

        Returns:
            -v: the negative value of the win/loss w.r.t. the current player
        """
        canonicalBoard = self.game.getCanonicalBoard(canonicalBoard, 1)  # inputs ignored --> ignore getCanonicalBoard(), its for neural network so we dont have to worry about using this
        s = self.game.stringRepresentation(canonicalBoard)
        if s in self.Es:
            # terminal node
            return -self.Es[s]

        if s not in self.Ps:
            # leaf node
            self.Ps[s], v = self.nnet.predict(canonicalBoard) # DONT NEED
            valids = self.game.getValidMoves(canonicalBoard, 1, search=False) # valid moves --> legal moves / all empty moves
            self.Vs[s] = valids
            self.Ns[s] = 0

            if self.Vs[s].max() == 0:   # terminal state --> just import python library
                self.Es[s] = -1
                return -self.Es[s]

            self.Ps[s] = self.Ps[s] * valids    # masking invalid moves, change to array, instead of a vector
                                                # this basically fills illegal moves with 0 so our simulation will never play at illegal moves
            sum_Ps_s = np.sum(self.Ps[s])
            if sum_Ps_s > 0:
                self.Ps[s] /= sum_Ps_s      # re-normalize
            else:
                print("All valid moves were masked by NN")
                self.Ps[s] = self.Ps[s] + valids
                self.Ps[s] /= np.sum(self.Ps[s])

            return -v

        best_a = -1
        best_u = -float("inf")
        # pick the action with the highest upper confidence bound
        # set p (probility vairable in PUCT) to 1 --> p == self.Ps[s][a]
        for a in range(self.game.getActionSize()): # getActionSize() just returns a number for all possible actions
            if self.Vs[s][a]:
                if (s,a) in self.Qsa:
                    self.Ps[s][a] = 1 # set to 1 so it does not affect the uct formula
                    u = self.Qsa[(s,a)] + self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (1+self.Nsa[(s,a)])
                else:
                    u = 1.0 * self.Ps[s][a] * math.sqrt(self.Ns[s]+EPS)     # Q = 0 ?

                if u > best_u:
                    best_u = u
                    best_a = a
        
        a = best_a
        assert a != -1

        next_s, next_player = self.game.getNextState(canonicalBoard, 1, a)
        
        v = self.search(next_s)

        if (s,a) in self.Qsa:
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
        self.Ps.clear()
        self.Vs.clear()
        self.Es.clear()
        return
