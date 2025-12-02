import math
import chess

# Reuse your existing engine as a base
from minimax_group.minimax_bot import MinimaxBot as BaseMinimaxBot


class FastMinimaxBot(BaseMinimaxBot):
    """
    Same as your current MinimaxBot, but with a cheaper quiescence search.
    - Adds a QS depth limit
    - Optional QS node limit
    - No SEE-based heavy pruning (just MVV-LVA ordering)
    """

    def __init__(self, depth=6, eval_fn=None,
                 qs_depth_limit=6,
                 qs_nodes_limit=50000):
        super().__init__(depth=depth, eval_fn=eval_fn)
        self.qs_depth_limit = qs_depth_limit
        self.qs_nodes_limit = qs_nodes_limit
        self._qs_nodes = 0

    def play(self, board: chess.Board):
        """Reset QS node counter each root search."""
        self._qs_nodes = 0
        return super().play(board)

    # ---- OVERRIDE ONLY QUIESCENCE ----
    def quiescence(self,
                   board: chess.Board,
                   alpha: float,
                   beta: float,
                   ply: int,
                   qs_depth: int = 0) -> float:
        # Safety 1: depth cap for QS
        if qs_depth >= self.qs_depth_limit:
            return self.eval_fn(board)

        # Safety 2: global QS node cap per search
        self._qs_nodes += 1
        if self._qs_nodes >= self.qs_nodes_limit:
            return self.eval_fn(board)

        # Stand-pat evaluation (static score)
        stand_pat = self.eval_fn(board)

        # Alpha–beta on stand-pat
        if board.turn == chess.WHITE:
            if stand_pat >= beta:
                return beta
            if stand_pat > alpha:
                alpha = stand_pat
        else:
            if stand_pat <= alpha:
                return alpha
            if stand_pat < beta:
                beta = stand_pat

        # Only consider noisy moves: captures or promotions
        moves = [m for m in board.legal_moves if board.is_capture(m) or m.promotion]
        if not moves:
            return stand_pat

        # Simple ordering: best victim–attacker first (reuse your victim_value)
        moves.sort(key=lambda m: self.victim_value(board, m), reverse=True)

        if board.turn == chess.WHITE:
            best = stand_pat
            for move in moves:
                board.push(move)
                score = self.quiescence(board, alpha, beta, ply + 1, qs_depth + 1)
                board.pop()

                if score > best:
                    best = score
                if score > alpha:
                    alpha = score
                if alpha >= beta:
                    return beta  # fail-hard cutoff
            return best
        else:
            best = stand_pat
            for move in moves:
                board.push(move)
                score = self.quiescence(board, alpha, beta, ply + 1, qs_depth + 1)
                board.pop()

                if score < best:
                    best = score
                if score < beta:
                    beta = score
                if beta <= alpha:
                    return alpha
            return best
