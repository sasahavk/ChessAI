import chess
import math, random

VAL_WIN = 9999999
VAL_LOSE = -9999999
VAL_TIE = 0


class Node:
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
        c = 1.41421356
        return (self.score / self.visits) + c * math.sqrt(math.log(self.parent.visits) / self.visits)

    def best_child(self):
        return max(self.children, key=lambda n: n.ucb1())

    def add_child(self, move):
        newBoard = self.board.copy()
        newBoard.push(move)
        child = Node(newBoard, parent=self, lastMove=move)
        self.untried_moves.remove(move)
        self.children.append(child)
        return child


def backupEvalFunc(board: chess.Board):
    vals = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
    }
    score = 0
    for piece, v in vals.items():
        score += v * (len(board.pieces(piece, chess.WHITE)) - len(board.pieces(piece, chess.BLACK)))
    return score


class MonteCarloSearchTreeBot:
    def __init__(self, numRootSimulations, maxSimDepth, evalFunc=None):
        self.numRootSimulations = numRootSimulations
        self.maxSimDepth = maxSimDepth
        self.evalFunc = backupEvalFunc if evalFunc is None else evalFunc
        self.root_player = chess.WHITE

    def play(self, board):
        self.root_player = board.turn
        root = Node(board)

        for _ in range(self.numRootSimulations):
            leaf = self.applyTreePolicy(root)
            result = self.rollout(leaf)
            self.backpropagate(leaf, result)

        if not root.children:
            return random.choice(list(board.legal_moves))

        return max(root.children, key=lambda n: n.score / n.visits).lastMove

    def applyTreePolicy(self, node):
        current = node
        while not current.board.is_game_over():
            if current.untried_moves:
                move = random.choice(current.untried_moves)
                return current.add_child(move)
            current = current.best_child()
        return current

    def rollout(self, node):
        simBoard = node.board.copy()
        for _ in range(self.maxSimDepth):
            if simBoard.is_game_over():
                result = simBoard.result()
                if result == "1-0":
                    return VAL_WIN if self.root_player == chess.WHITE else VAL_LOSE
                elif result == "0-1":
                    return VAL_LOSE if self.root_player == chess.WHITE else VAL_WIN
                else:
                    return VAL_TIE
            moves = list(simBoard.legal_moves)
            if not moves:
                break
            simBoard.push(random.choice(moves))

        raw = self.evalFunc(simBoard)
        return raw if self.root_player == chess.WHITE else -raw

    def backpropagate(self, node, score):
        current = node
        while current is not None:
            current.visits += 1
            current.score += score
            current = current.parent


def mcts_search(board: chess.Board, iters=600, max_depth=40):
    bot = MonteCarloSearchTreeBot(numRootSimulations=iters, maxSimDepth=max_depth)
    return bot.play(board)
