import chess
import math

MVV = {
    chess.PAWN: 100, chess.KNIGHT: 300, chess.BISHOP: 300,
    chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 10000
}

# For SEE we can reuse MVV values; split out for clarity
SEE_VAL = {
    chess.PAWN: 100, chess.KNIGHT: 300, chess.BISHOP: 300,
    chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 10000
}

# Skip captures in QS if SEE says you lose ≥ this many centipawns
SEE_PRUNE_MARGIN = 50  # 0.5 pawn; tune 0..100


MATE_SCORE = 1000000

class MinimaxBot:
    def __init__(self, depth=6, eval_fn=None):
        self.depth = depth
        self.eval_fn = eval_fn  # evaluate(board) returns + for white, - for black

    def play(self, board):
        """
        Returns the best move for the current board position.
        White always maximizes.
        Black always minimizes.
        """
        if board.is_game_over():
            return None

        alpha = -math.inf
        beta = math.inf
        best_move = None

        # White maximizes score, Black minimizes
        best_value = -math.inf if board.turn == chess.WHITE else math.inf

        legal_moves = list(board.legal_moves)
        legal_moves.sort(key=lambda m: self.order_key(board, m), reverse=True)

        for move in legal_moves:
            board.push(move)
            value = self.minimax(board, self.depth - 1, alpha, beta, ply = 1)
            board.pop()
            #print("Move: ", move)
            #print("Value: ", value)
            if board.turn == chess.WHITE:
                # We restored the board, so board.turn is the side BEFORE move
                # That means we check white choice here
                if value > best_value:
                    best_value = value
                    best_move = move
                alpha = max(alpha, best_value)
            else:
                if value < best_value:
                    best_value = value
                    best_move = move
                beta = min(beta, best_value)

            if beta <= alpha:
                break
        #print("\nBest move:", best_move)
        #print("Best score:", best_value)
        return best_move

    def minimax(self, board, depth, alpha, beta, ply):
        """
        Minimax using board.turn instead of passing maximizing_player.
        eval_fn returns score from WHITE perspective.
        """
        # Terminal states
        if depth == 0 or board.is_game_over(claim_draw=False):

            # Checkmate scoring (from white POV)
            if board.is_checkmate():
                return -(MATE_SCORE - ply) if board.turn == chess.WHITE else (MATE_SCORE - ply)

            # Draw
            if board.is_stalemate() or board.is_insufficient_material():
                return 0


            if board.is_check():
                depth = 1  # extend once for checking nodes
            else:
                return self.quiescence(board, alpha, beta, ply)

            #print(self.eval_fn(board))
            #return self.eval_fn(board)  # No sign flipping

        # White to move → maximize score
        if board.turn == chess.WHITE:
            max_eval = -math.inf
            legal_moves = list(board.legal_moves)
            legal_moves.sort(key=lambda m: self.order_key(board, m), reverse=True)
            for move in legal_moves:
                board.push(move)
                eval_value = self.minimax(board, depth - 1, alpha, beta, ply+1)
                board.pop()
                max_eval = max(max_eval, eval_value)
                alpha = max(alpha, eval_value)
                if beta <= alpha:
                    break
            return max_eval

        # Black to move → minimize score
        else:
            min_eval = math.inf
            legal_moves = list(board.legal_moves)
            legal_moves.sort(key=lambda m: self.order_key(board, m), reverse=True)
            for move in legal_moves:
                board.push(move)
                eval_value = self.minimax(board, depth - 1, alpha, beta, ply+1)
                board.pop()
                min_eval = min(min_eval, eval_value)
                beta = min(beta, eval_value)
                if beta <= alpha:
                    break
            return min_eval

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
        # Stand-pat (static) eval — assume no more tactics
        stand_pat = self.eval_fn(board)

        # Alpha-beta on stand-pat
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

        # Generate "noisy" moves: captures or promotions
        raw_moves = [m for m in board.legal_moves if board.is_capture(m) or m.promotion]
        if not raw_moves:
            return stand_pat

        # ---- NEW: SEE filter + ordering ----
        scored = []
        for m in raw_moves:
            see = self.static_exchange_eval(board, m)
            # prune clearly losing captures (tunable margin)
            if see < -SEE_PRUNE_MARGIN:
                continue
            # tie-break with your victim_value (MVV-LVA) to help αβ
            scored.append((see, self.victim_value(board, m), m))

        if not scored:
            return stand_pat

        # Order best tactical payoffs first
        scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
        moves = [m for _, __, m in scored]
        # ---- END NEW ----

        if board.turn == chess.WHITE:
            best = stand_pat
            for move in moves:
                board.push(move)
                score = self.quiescence(board, alpha, beta, ply + 1)
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
                score = self.quiescence(board, alpha, beta, ply + 1)
                board.pop()

                if score < best:
                    best = score
                if score < beta:
                    beta = score
                if beta <= alpha:
                    return alpha
            return best
