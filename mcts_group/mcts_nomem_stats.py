import chess
import math, random
from minimax_group_evaluate import evaluate as minimax_evaluate

VAL_WIN = 9999999
VAL_LOSE = -9999999
VAL_TIE = 0

class StatsNode:
    def __init__(self, board: chess.Board, parent=None, lastMove=None):
        self.board = board
        self.parent = parent
        self.children = []
        self.score = 0.0
        self.visits = 0
        self.lastMove = lastMove
        self.untried_moves = list(board.legal_moves)

    def ucb1(self):
        if self.visits == 0:
            return float("inf")
        return (self.score / self.visits) + 1.41421356 * math.sqrt(math.log(self.parent.visits) / self.visits)

    def best_child(self):
        return max(self.children, key=lambda n: n.ucb1())

    def add_child(self, move):
        newBoard = self.board.copy()
        newBoard.push(move)
        child = StatsNode(newBoard, parent=self, lastMove=move)
        self.untried_moves.remove(move)
        self.children.append(child)
        return child

class StatsMCTSBot:
    def __init__(self, numRootSimulations, maxSimDepth, evalFunc=None):
        self.numRootSimulations = numRootSimulations
        self.maxSimDepth = maxSimDepth
        self.evalFunc = minimax_evaluate if (evalFunc is None) else evalFunc
        self.root_player = None

    def play(self, board: chess.Board):
        self.root_player = board.turn
        root = StatsNode(board)
        sim_depths = []
        for _ in range(self.numRootSimulations):
            leaf = self.applyTreePolicy(root)
            score, depth = self.rollout_with_depth(leaf)
            if depth is not None:
                sim_depths.append(depth)
            self.backpropagate(leaf, score)
        if not root.children:
            move = random.choice(list(board.legal_moves))
            child_values = []
            avg_non_win = 0.0
        else:
            child_values = [(c.score / c.visits) for c in root.children if c.visits > 0]
            if child_values:
                best_value = max(child_values)
                non_best = [v for v in child_values if v < best_value]
                avg_non_win = sum(non_best) / len(non_best) if non_best else 0.0
            else:
                avg_non_win = 0.0
            best_child = max(root.children, key=lambda n: n.score / n.visits)
            move = best_child.lastMove
        avg_sim_depth = sum(sim_depths) / len(sim_depths) if sim_depths else 0.0
        stats = {
            "avg_sim_depth": avg_sim_depth,
            "root_simulations": self.numRootSimulations,
            "avg_non_win_eval": avg_non_win,
            "child_values": child_values
        }
        return move, stats

    def applyTreePolicy(self, node: StatsNode):
        current = node
        while not current.board.is_game_over():
            if current.untried_moves or not current.children:
                move = random.choice(current.untried_moves)
                return current.add_child(move)
            current = current.best_child()
        return current

    def rollout_with_depth(self, node: StatsNode):
        simBoard = node.board.copy()
        depth = 0
        for depth in range(self.maxSimDepth):
            if simBoard.is_game_over():
                result = simBoard.result()
                if result == "1-0":
                    return (VAL_WIN if self.root_player == chess.WHITE else VAL_LOSE, depth)
                elif result == "0-1":
                    return (VAL_LOSE if self.root_player == chess.WHITE else VAL_WIN, depth)
                else:
                    return (VAL_TIE, depth)
            moves = list(simBoard.legal_moves)
            if not moves:
                break
            simBoard.push(random.choice(moves))
        raw = self.evalFunc(simBoard)
        score = raw if self.root_player == chess.WHITE else -raw
        return score, depth

    def backpropagate(self, node: StatsNode, score: int):
        current = node
        while current is not None:
            current.visits += 1
            current.score += score
            current = current.parent
