import chess

# === Helper bitboard shift functions ===
def shift_up_left(bb): return (bb << 7) & ~chess.BB_FILE_H
def shift_up_right(bb): return (bb << 9) & ~chess.BB_FILE_A
def shift_down_left(bb): return (bb >> 9) & ~chess.BB_FILE_H
def shift_down_right(bb): return (bb >> 7) & ~chess.BB_FILE_A

# === Piece values ===
piece_values = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}

# === Piece-square tables (White's POV) ===
pawn_table = [
      0,  0,  0,  0,  0,  0,  0,  0,
     50, 50, 50, 50, 50, 50, 50, 50,
     10, 10, 20, 30, 30, 20, 10, 10,
      5,  5, 10, 25, 25, 10,  5,  5,
      0,  0,  0, 20, 20,  0,  0,  0,
      5, -5,-10,  0,  0,-10, -5,  5,
      5, 10, 10,-22,-22, 10, 10,  5,
      0,  0,  0,  0,  0,  0,  0,  0
]
knight_table = [
    -50,-30,-30,-30,-30,-30,-30,-50,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -50,-30,-30,-30,-30,-30,-30,-50
]
bishop_table = [
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5, 10, 10,  5,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -20,-10,-10,-10,-10,-10,-10,-20
]
rook_table = [
      0,  0,  0,  0,  0,  0,  0,  0,
      5, 10, 10, 10, 10, 10, 10,  5,
     -5,  0,  0,  0,  0,  0,  0, -5,
     -5,  0,  0,  0,  0,  0,  0, -5,
     -5,  0,  0,  0,  0,  0,  0, -5,
     -5,  0,  0,  0,  0,  0,  0, -5,
     -5,  0,  0,  0,  0,  0,  0, -5,
      0,  0,  0,  5,  5,  0,  0,  0
]
queen_table = [
    -20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5,  5,  5,  5,  0,-10,
     -5,  0,  5,  5,  5,  5,  0, -5,
      0,  0,  5,  5,  5,  5,  0, -5,
    -10,  5,  5,  5,  5,  5,  0,-10,
    -10,  0,  5,  0,  0,  0,  0,-10,
    -20,-10,-10, -5, -5,-10,-10,-20
]
king_table = [
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -10,-20,-20,-20,-20,-20,-20,-10,
     20, 20,  0,  0,  0,  0, 20, 20,
     20, 30, 10,  0,  0, 10, 30, 20
]

piece_square_tables = {
    chess.PAWN: pawn_table,
    chess.KNIGHT: knight_table,
    chess.BISHOP: bishop_table,
    chess.ROOK: rook_table,
    chess.QUEEN: queen_table,
    chess.KING: king_table
}

# === Evaluation components ===
def evaluate_material(board: chess.Board) -> int:
    score = 0
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece:
            val = piece_values[piece.piece_type]
            score += val if piece.color == chess.WHITE else -val
    return score

def evaluate_piece_square_tables(board: chess.Board) -> int:
    score = 0
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece:
            pst = piece_square_tables[piece.piece_type]
            if piece.color == chess.WHITE:
                score += pst[sq]
            else:
                score -= pst[chess.square_mirror(sq)]
    return score

def evaluate_pawn_structure(board: chess.Board) -> int:
    score = 0
    for color in (chess.WHITE, chess.BLACK):
        pawns = board.pieces(chess.PAWN, color)
        files = [pawns & chess.BB_FILES[f] for f in range(8)]

        # doubled pawns
        for f in range(8):
            n = files[f].bit_count()
            if n > 1:
                score += (-15 * (n - 1)) if color == chess.WHITE else (15 * (n - 1))

        # isolated pawns
        for f in range(8):
            if files[f]:
                left = files[f - 1] if f > 0 else 0
                right = files[f + 1] if f < 7 else 0
                if (left | right) == 0:
                    score += (-20) if color == chess.WHITE else 20

        # passed pawns
        enemy_pawns = board.pieces(chess.PAWN, not color)
        for sq in pawns:
            f = chess.square_file(sq)
            r = chess.square_rank(sq)
            mask = 0
            for df in (-1, 0, 1):
                nf = f + df
                if 0 <= nf <= 7:
                    if color == chess.WHITE:
                        mask |= chess.BB_FILES[nf] & sum(chess.BB_RANKS[r+1:], 0)
                    else:
                        mask |= chess.BB_FILES[nf] & sum(chess.BB_RANKS[:r], 0)
            passed = not (enemy_pawns & mask)
            if passed:
                bonus = 8 * (r if color == chess.WHITE else 7 - r)
                score += bonus if color == chess.WHITE else -bonus
    return score

def evaluate_bishop_pair(board):
    score = 0
    for color in [chess.WHITE, chess.BLACK]:
        if len(board.pieces(chess.BISHOP, color)) >= 2:
            score += 30 if color == chess.WHITE else -30
    return score

def evaluate_rook_files(board):
    score = 0
    for color in [chess.WHITE, chess.BLACK]:
        rooks = board.pieces(chess.ROOK, color)
        for r_sq in rooks:
            f = chess.square_file(r_sq)
            enemy_pawns = [p for p in board.pieces(chess.PAWN, not color) if chess.square_file(p) == f]
            friendly_pawns = [p for p in board.pieces(chess.PAWN, color) if chess.square_file(p) == f]
            if not enemy_pawns and not friendly_pawns:
                score += 25 if color == chess.WHITE else -25
            elif not friendly_pawns and enemy_pawns:
                score += 15 if color == chess.WHITE else -15
    return score

def evaluate_knight_outposts(board):
    score = 0
    white_pawn_attacks = shift_up_left(board.pieces(chess.PAWN, chess.WHITE)) | shift_up_right(board.pieces(chess.PAWN, chess.WHITE))
    black_pawn_attacks = shift_down_left(board.pieces(chess.PAWN, chess.BLACK)) | shift_down_right(board.pieces(chess.PAWN, chess.BLACK))
    for color in (chess.WHITE, chess.BLACK):
        knights = board.pieces(chess.KNIGHT, color)
        if color == chess.WHITE:
            territory_mask = chess.BB_RANK_4 | chess.BB_RANK_5 | chess.BB_RANK_6 | chess.BB_RANK_7
            attacked_by_enemy = black_pawn_attacks
        else:
            territory_mask = chess.BB_RANK_1 | chess.BB_RANK_2 | chess.BB_RANK_3 | chess.BB_RANK_4
            attacked_by_enemy = white_pawn_attacks
        outposts = (knights & territory_mask) & ~attacked_by_enemy
        count = outposts.bit_count()
        score += (20 * count) if color == chess.WHITE else -(20 * count)
    return score

def game_phase(board):
    max_phase = 2*320 + 2*330 + 2*500 + 2*900
    current = sum(piece_values[t] * (len(board.pieces(t, c)) + len(board.pieces(t, not c)))
                  for t in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN] for c in [chess.WHITE])
    phase = current / max_phase
    return max(0.0, min(1.0, phase))

def evaluate_king(board):
    phase = game_phase(board)
    score = 0
    for color in [chess.WHITE, chess.BLACK]:
        kings = list(board.pieces(chess.KING, color))
        if not kings:
            continue
        king_sq = kings[0]
        kr, kf = chess.square_rank(king_sq), chess.square_file(king_sq)
        safety_score = 0
        pawn_dir = 1 if color == chess.WHITE else -1
        ramp_factor = min(board.fullmove_number / 5, 1.0)
        for df in [-1, 0, 1]:
            r, f = kr + pawn_dir, kf + df
            if 0 <= r <= 7 and 0 <= f <= 7:
                p = board.piece_at(chess.square(f, r))
                if p and p.piece_type == chess.PAWN and p.color == color:
                    safety_score += 10 * ramp_factor
        danger = 0
        for sq in chess.SQUARES:
            p = board.piece_at(sq)
            if p and p.color != color:
                dist = abs(kf - chess.square_file(sq)) + abs(kr - chess.square_rank(sq))
                if dist <= 2:
                    danger += 5 if p.piece_type != chess.PAWN else 2
        safety_score -= danger
        centralization = -(abs(kf - 3.5) + abs(kr - 3.5)) * 10
        king_score = phase * safety_score + (1 - phase) * centralization
        score += king_score if color == chess.WHITE else -king_score
    return score

def evaluate_mobility(board):
    white, black = 0, 0
    center = [chess.D4, chess.E4, chess.D5, chess.E5]
    for move in board.legal_moves:
        piece = board.piece_at(move.from_square)
        if not piece:
            continue
        weight = 0
        if piece.piece_type == chess.PAWN:
            rank = chess.square_rank(move.to_square)
            weight = 7 + (rank * 2 if piece.color == chess.WHITE else (7 - rank) * 2)
            if move.to_square in center:
                weight += 15
        elif piece.piece_type == chess.KNIGHT:
            weight = 4 - (5 if board.fullmove_number == 1 else 0)
        elif piece.piece_type in (chess.BISHOP, chess.ROOK):
            weight = 6
        elif piece.piece_type == chess.QUEEN:
            weight = 10
        (white if piece.color == chess.WHITE else black).__iadd__(weight)
    return white - black

def evaluate_center_control(board):
    score = 0
    center = [chess.E4, chess.D4, chess.E5, chess.D5]
    for sq in center:
        piece = board.piece_at(sq)
        if piece:
            val = 80 if piece.piece_type == chess.PAWN else 15
            score += val if piece.color == chess.WHITE else -val
    return score

def evaluate(board):
    phase_value = game_phase(board)
    w_material = 1.0
    w_pst = 0.3 + 0.7 * phase_value
    w_pawn_structure = 0.4
    w_bishop_pair = 0.2
    w_knight_outposts = 0.3
    w_rook_files = 0.2
    w_king_safety = 0.6
    w_mobility = 0.3
    w_center = 0.38

    score = 0
    score += w_material * evaluate_material(board)
    score += w_pst * evaluate_piece_square_tables(board)
    score += w_pawn_structure * evaluate_pawn_structure(board)
    score += w_bishop_pair * evaluate_bishop_pair(board)
    score += w_knight_outposts * evaluate_knight_outposts(board)
    score += w_rook_files * evaluate_rook_files(board)
    score += w_king_safety * evaluate_king(board)
    score += w_mobility * phase_value * evaluate_mobility(board)
    score += w_center * evaluate_center_control(board)
    score += 10 if board.turn == chess.WHITE else -10
    return int(score)
