import chess
import math
from dataclasses import dataclass
from enum import Enum, auto
import hashlib
from chess.polyglot import zobrist_hash as poly_zobrist_hash


class TTFlag(Enum):
    EXACT = auto()
    LOWERBOUND = auto()
    UPPERBOUND = auto()


@dataclass
class TTEntry:
    depth: int
    value: float
    flag: TTFlag
    move: chess.Move | None


TT_MAX = 200000

MVV = {
    chess.PAWN: 100, chess.KNIGHT: 300, chess.BISHOP: 300,
    chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 10000
}

# For SEE, reuse MVV values; split out for clarity
SEE_VAL = {
    chess.PAWN: 100, chess.KNIGHT: 300, chess.BISHOP: 300,
    chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 10000
}

# Skip captures in QS if SEE says you lose ≥ this many centipawns
SEE_PRUNE_MARGIN = 50  # 0.5 pawn; tune 0..100

MATE_SCORE = 1000000


class NegamaxBot:
    def __init__(self, depth=6, eval_fn=None):
        self.depth = depth
        self.eval_fn = eval_fn  # evaluate(board) returns + for white, - for black
        self.tt: dict[int, TTEntry] = {}

    def preferred_first(self, moves: list[chess.Move], preferred: chess.Move | None):
        if preferred is None:
            return
        try:
            i = moves.index(preferred)
            if i != 0:
                moves[0], moves[i] = moves[i], moves[0]
        except ValueError:
            pass

    def order_with_tt_and_pv(self, moves, tt_move, pv_move):
        """Prioritize TT move first, then PV move (if different)."""
        if tt_move and tt_move in moves:
            moves.remove(tt_move)
            moves.insert(0, tt_move)
        if pv_move and pv_move != tt_move and pv_move in moves:
            moves.remove(pv_move)
            moves.insert(1, pv_move)

    def tt_probe(self, board: chess.Board, depth: int, alpha: float, beta: float):
        key = poly_zobrist_hash(board) & 0xFFFFFFFFFFFFFFFF
        ent = self.tt.get(key)
        if ent is None:
            return None, None
        if ent.depth >= depth:
            if ent.flag == TTFlag.EXACT:
                return ent.value, ent.move
            if ent.flag == TTFlag.LOWERBOUND and ent.value >= beta:
                return ent.value, ent.move
            if ent.flag == TTFlag.UPPERBOUND and ent.value <= alpha:
                return ent.value, ent.move
        return None, ent.move

    def tt_store(self, board: chess.Board, depth: int, value: float,
                 alpha_orig: float, beta_orig: float, best_move: chess.Move | None):
        key = poly_zobrist_hash(board) & 0xFFFFFFFFFFFFFFFF
        if value <= alpha_orig:
            flag = TTFlag.UPPERBOUND
        elif value >= beta_orig:
            flag = TTFlag.LOWERBOUND
        else:
            flag = TTFlag.EXACT

        prev = self.tt.get(key)
        if (prev is None) or (depth >= prev.depth):
            if len(self.tt) >= TT_MAX:
                self.tt.pop(next(iter(self.tt)))
            self.tt[key] = TTEntry(depth=depth, value=value, flag=flag, move=best_move)

    def _eval_stm(self, board: chess.Board) -> float:
        """
        Evaluation from side-to-move perspective.
        eval_fn gives + for white, - for black.
        """
        raw = self.eval_fn(board)
        return raw if board.turn == chess.WHITE else -raw

    def play(self, board):
        if board.is_game_over():
            return None

        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None

        # Initial ordering
        legal_moves.sort(key=lambda m: self.order_key(board, m), reverse=True)

        pv_move = None
        best_move_overall = legal_moves[0]  # fallback if nothing improves

        for depth in range(1, self.depth + 1):
            alpha = -math.inf
            beta = math.inf
            best_move_this_depth = None
            best_value_this_depth = -math.inf

            # Order for this iteration: TT move then PV move
            ordered_moves = legal_moves.copy()
            tt_val, tt_move = self.tt_probe(board, depth, alpha, beta)
            self.order_with_tt_and_pv(ordered_moves, tt_move, pv_move)

            for move in ordered_moves:
                board.push(move)
                # Negamax at root: value from root side's perspective
                value = -self.negamax(board, depth - 1, -beta, -alpha, ply=1)
                board.pop()

                if value > best_value_this_depth:
                    best_value_this_depth = value
                    best_move_this_depth = move

                if value > alpha:
                    alpha = value

                if alpha >= beta:
                    break

            # Finalize PV and best for this depth
            if best_move_this_depth is not None:
                best_move_overall = best_move_this_depth
                pv_move = best_move_this_depth  # used to order next depth

        return best_move_overall

    def negamax(self, board: chess.Board, depth: int, alpha: float, beta: float, ply: int) -> float:
        # Terminal states (drawish)
        if board.is_stalemate() or board.is_insufficient_material():
            return 0

        if board.can_claim_fifty_moves() or board.can_claim_threefold_repetition():
            return 0

        # Full terminal check
        if board.is_game_over(claim_draw=False):
            if board.is_checkmate():
                # Side to move is checkmated → losing from their perspective
                return -(MATE_SCORE - ply)
            return 0

        # --- depth / QS handoff ---
        if depth <= 0:
            if board.is_check():
                depth = 1  # one-ply check extension
            else:
                return self.quiescence(board, alpha, beta, ply)

        alpha_orig, beta_orig = alpha, beta

        # Transposition table probe
        tt_value, tt_move = self.tt_probe(board, depth, alpha, beta)
        if tt_value is not None:
            return tt_value

        best_val = -math.inf
        best_move = None

        legal_moves = list(board.legal_moves)
        legal_moves.sort(key=lambda m: self.order_key(board, m), reverse=True)
        self.preferred_first(legal_moves, tt_move)

        for move in legal_moves:
            board.push(move)
            val = -self.negamax(board, depth - 1, -beta, -alpha, ply + 1)
            board.pop()

            if val > best_val:
                best_val = val
                best_move = move

            if val > alpha:
                alpha = val

            if alpha >= beta:
                break

        self.tt_store(board, depth, best_val, alpha_orig, beta_orig, best_move)
        return best_val

    """def _store_killer(self, ply: int, move: chess.Move):
        k1, k2 = self.killers.get(ply, (None, None))
        if move != k1:
            self.killers[ply] = (move, k1)"""

    def victim_value(self, board: chess.Board, move: chess.Move) -> int:
        if not board.is_capture(move):
            return 0

        victim = board.piece_at(move.to_square)
        attacker = board.piece_at(move.from_square)

        if victim is None and board.is_en_passant(move):
            victim_val = MVV[chess.PAWN]
        else:
            victim_val = MVV.get(victim.piece_type, 0) if victim else 0

        attacker_val = MVV.get(attacker.piece_type, 0) if attacker else 0

        return victim_val - attacker_val

    def order_key(self, board: chess.Board, move: chess.Move) -> tuple:
        is_cap = board.is_capture(move)
        vv = self.victim_value(board, move)
        promo = 1 if move.promotion else 0
        gives_check = 1 if board.gives_check(move) else 0

        """killer_bonus = 0
        if killer1 and move == killer1:
            killer_bonus = 1
        elif killer2 and move == killer2:
            killer_bonus = 1

        hist_bonus = 0
        if history is not None:
            hist_bonus = history.get((board.turn, move.from_square, move.to_square), 0)"""

        # Sort high -> low
        return (is_cap, vv + promo * 50, gives_check)

    def _see_piece_val(self, board: chess.Board, sq: int) -> int:
        p = board.piece_at(sq)
        return SEE_VAL.get(p.piece_type, 0) if p else 0

    def _see_promo_delta(self, move: chess.Move) -> int:
        # Value gained from promoting (piece gained minus pawn)
        if move.promotion:
            return SEE_VAL[move.promotion] - SEE_VAL[chess.PAWN]
        return 0

    def static_exchange_eval(self, board: chess.Board, move: chess.Move) -> int:
        """
        SEE: estimate the net material result (centipawns) from doing `move`
        and then allowing optimal alternating recaptures on the target square.
        Positive = good for side to move; negative = bad.
        Designed for pruning/ordering captures in QS.
        """
        target = move.to_square

        # First “gain” is the victim on target (handle EP specially)
        tmp = board.copy(stack=False)

        if tmp.is_en_passant(move):
            victim_value = SEE_VAL[chess.PAWN]
        else:
            victim_value = self._see_piece_val(tmp, target)

        promo_gain = self._see_promo_delta(move)
        total = victim_value + promo_gain  # original side's initial gain

        # Make the capture
        tmp.push(move)

        # After pushing, it's the opponent's turn
        sign = -1  # subtract opponent gains from our perspective

        while True:
            # List all *legal* recaptures that land back on `target`
            candidates = []
            for m in tmp.legal_moves:
                # must land on target and be a capture
                if m.to_square == target and tmp.is_capture(m):
                    attacker = tmp.piece_at(m.from_square)
                    if attacker is None:
                        continue
                    attacker_val = SEE_VAL.get(attacker.piece_type, 0)
                    promo_delta = self._see_promo_delta(m)
                    candidates.append((attacker_val, promo_delta, m))

            if not candidates:
                break

            # Least Valuable Attacker (LVA) recaptures first
            attacker_val, promo_delta, recap = min(candidates, key=lambda x: x[0])

            # Opponent “gains” our piece on target → from our POV, subtract (sign = -1 here)
            total += sign * (attacker_val + promo_delta)

            # Play the recapture and alternate side
            tmp.push(recap)
            sign *= -1

        return total

    def quiescence(self, board: chess.Board, alpha: float, beta: float, ply: int) -> float:
        # Stand-pat (static) eval — from side-to-move perspective
        stand_pat = self._eval_stm(board)

        # Alpha-beta on stand-pat
        if stand_pat >= beta:
            return beta
        if stand_pat > alpha:
            alpha = stand_pat

        # Generate "noisy" moves: captures or promotions
        raw_moves = [m for m in board.legal_moves if board.is_capture(m) or m.promotion]
        if not raw_moves:
            return stand_pat

        # ---- SEE filter + ordering ----
        buckets = []
        for m in raw_moves:
            mvv_lva = self.victim_value(board, m)  # (victim-attacker)
            # obvious losers: skip immediately
            if mvv_lva < -150:  # tune: -100..-200
                continue
            # obvious winners: trust MVV–LVA, no SEE needed
            if mvv_lva > 150:
                buckets.append((9999, mvv_lva, m))  # huge SEE key so they sort first
                continue
            # marginal: compute SEE (expensive)
            see = self.static_exchange_eval(board, m)
            if see < -SEE_PRUNE_MARGIN:  # prune small losers
                continue
            buckets.append((see, mvv_lva, m))

        if not buckets:
            return stand_pat

        # order by (SEE or 9999), then MVV–LVA
        buckets.sort(key=lambda t: (t[0], t[1]), reverse=True)
        moves = [m for _, __, m in buckets]

        best = stand_pat
        for move in moves:
            board.push(move)
            score = -self.quiescence(board, -beta, -alpha, ply + 1)
            board.pop()

            if score > best:
                best = score
            if score > alpha:
                alpha = score
            if alpha >= beta:
                return beta  # fail-hard cutoff

        return best