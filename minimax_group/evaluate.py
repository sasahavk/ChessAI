import chess

# NOTE: Weights here are NOT fully fine tuned - minimal testing
# These represent the traditonal centipawn values of the pieces in chess
piece_values = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,  # Older models used to use 300 for both Knights/Bishops
    chess.BISHOP: 330,  # but 320/330 split between these is now the common approach
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,  # King does not have an associated centipawn value
}

# Pawn PST
pawn_table = [
    # 8x8
    0,   0,   0,   0,   0,   0,   0,   0,
    50, 50,  50,  50,  50,  50,  50,  50,
    10, 10,  20,  30,  30,  20,  10,  10,
    5,   5,  10,  27,  27,  10,   5,   5,
    0,   0,   0,  25,  25,   0,   0,   0,
    5,  -5, -10,   0,   0, -10,  -5,   5,
    5,  10,  10, -25, -25,  10,  10,   5,
    0,   0,   0,   0,   0,   0,   0,   0
]

# Knight PST
knight_table = [
    -50, -40, -30, -30, -30, -30, -40, -50,
    -40, -20,   0,   5,   5,   0, -20, -40,
    -30,   5,  10,  15,  15,  10,   5, -30,
    -30,   0,  15,  20,  20,  15,   0, -30,
    -30,   5,  15,  20,  20,  15,   5, -30,
    -30,   0,  20,  15,  15,  20,   0, -30,
    -40, -20,   0,   0,   0,   0, -20, -40,
    -50, -5, -30, -30, -30, -30, -5, -50
]

# Bishop PST
bishop_table = [
    -20, -10, -10, -10, -10, -10, -10, -20,
    -10,   0,   0,   0,   0,   0,   0, -10,
    -10,   0,   5,  10,  10,   5,   0, -10,
    -10,   5,   5,  10,  10,   5,   5, -10,
    -10,   0,  10,  10,  10,  10,   0, -10,
    -10,  10,  10,  10,  10,  10,  10, -10,
    -10,   5,   0,   0,   0,   0,   5, -10,
    -20, -10, -10, -10, -10, -10, -10, -20
]

# Rook PST
rook_table = [
     0,   0,   0,   0,   0,   0,   0,   0,
     5,  10,  10,  10,  10,  10,  10,   5,
    -5,   0,   0,   0,   0,   0,   0,  -5,
    -5,   0,   0,   0,   0,   0,   0,  -5,
    -5,   0,   0,   0,   0,   0,   0,  -5,
    -5,   0,   0,   0,   0,   0,   0,  -5,
    -5,   0,   0,   0,   0,   0,   0,  -5,
     0,   0,   0,   5,   5,   0,   0,   0
]

# Queen PST
queen_table = [
    -20, -10, -10,  -5,  -5, -10, -10, -20,
    -10,   0,   0,   0,   0,   0,   0, -10,
    -10,   0,   5,   5,   5,   5,   0, -10,
     -5,   0,   5,   5,   5,   5,   0,  -5,
      0,   0,   5,   5,   5,   5,   0,  -5,
    -10,   5,   5,   5,   5,   5,   0, -10,
    -10,   0,   5,   0,   0,   0,   0, -10,
    -20, -10, -10,  -5,  -5, -10, -10, -20
]

# King PST
king_table = [
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -20, -30, -30, -40, -40, -30, -30, -20,
    -10, -20, -20, -20, -20, -20, -20, -10,
     20,  20,   0,   0,   0,   0,  20,  20,
     20,  30,  10,   0,   0,  10,  30,  20
]

# Combine into one dict
piece_square_tables = {
    chess.PAWN: pawn_table,
    chess.KNIGHT: knight_table,
    chess.BISHOP: bishop_table,
    chess.ROOK: rook_table,
    chess.QUEEN: queen_table,
    chess.KING: king_table
}


def evaluate_material(
        board: chess.Board) -> int:  # This function calculates the material on the board and returns total score
    score = 0
    for square in chess.SQUARES:  # Check each square on the board
        piece = board.piece_at(square)
        if piece:  # If there is a piece on that square
            piece_value = piece_values[piece.piece_type]
            if piece.color == chess.WHITE:
                score += piece_value  # Add the value of the piece if it belongs to white
            else:  # Piece belongs to black
                score -= piece_value  # Subtract the value of the piece since not white's

    return score


def evaluate_piece_square_tables(board: chess.Board) -> int:
    score = 0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            pst = piece_square_tables[piece.piece_type]

            # For White, use square as-is; for Black, mirror vertically
            if piece.color == chess.WHITE:
                score += pst[square]
            else:
                mirrored_square = chess.square_mirror(square)
                score -= pst[mirrored_square]
    return score


def evaluate_pawn_structure(board):
    score = 0

    for color in [chess.WHITE, chess.BLACK]:
        pawns = list(board.pieces(chess.PAWN, color))
        enemy_pawns = list(board.pieces(chess.PAWN, not color))
        files = {f: [] for f in range(8)}

        for sq in pawns:
            files[chess.square_file(sq)].append(sq)

        # -------------------
        # Doubled pawns
        # -------------------
        for f in range(8):
            n = len(files[f])
            if n > 1:
                score += (-15 * (n - 1)) if color == chess.WHITE else (15 * (n - 1))

        # -------------------
        # Isolated pawns
        # -------------------
        for f in range(8):
            if len(files[f]) == 0:
                continue
            left = files[f-1] if f > 0 else []
            right = files[f+1] if f < 7 else []
            if len(left) == 0 and len(right) == 0:
                score += -20 if color == chess.WHITE else 20

        # -------------------
        # Passed pawns
        # -------------------
        for sq in pawns:
            f = chess.square_file(sq)
            r = chess.square_rank(sq)
            passed = True

            for ep in enemy_pawns:
                ef = chess.square_file(ep)
                er = chess.square_rank(ep)
                if abs(ef - f) <= 1:
                    if (color == chess.WHITE and er > r) or (color == chess.BLACK and er < r):
                        passed = False
                        break

            if passed:
                advance = r if color == chess.WHITE else (7 - r)
                bonus = 8 * advance
                score += bonus if color == chess.WHITE else -bonus

        # -------------------
        # Hanging pawns (fixed)
        # -------------------
        for i in range(7):
            if len(files[i]) > 0 and len(files[i+1]) > 0:
                left_has = (i > 0 and len(files[i-1]) > 0)
                right_has = (i+2 < 8 and len(files[i+2]) > 0)
                if not left_has and not right_has:
                    score += -15 if color == chess.WHITE else 15

    return score

def evaluate_bishop_pair(board: chess.Board) -> int:
    score = 0
    for color in [chess.WHITE, chess.BLACK]:
        bishops = list(board.pieces(chess.BISHOP, color))
        if len(bishops) >= 2:
            score += 30 if color == chess.WHITE else -30
    return score


def evaluate_rook_files(board: chess.Board) -> int:
    score = 0
    for color in [chess.WHITE, chess.BLACK]:
        rooks = list(board.pieces(chess.ROOK, color))
        enemy_pawns = list(board.pieces(chess.PAWN, not color))
        friendly_pawns = list(board.pieces(chess.PAWN, color))
        for r_sq in rooks:
            f = chess.square_file(r_sq)
            # Pawns on this file
            enemy_file_pawns = [p for p in enemy_pawns if chess.square_file(p) == f]
            friendly_file_pawns = [p for p in friendly_pawns if chess.square_file(p) == f]

            # Open file: no pawns at all
            if len(enemy_file_pawns) == 0 and len(friendly_file_pawns) == 0:
                score += 25 if color == chess.WHITE else -25

            # Half-open file: no friendly pawns, at least one enemy pawn
            elif len(friendly_file_pawns) == 0 and len(enemy_file_pawns) > 0:
                score += 15 if color == chess.WHITE else -15

    return score


def evaluate_knight_outposts(board: chess.Board) -> int:
    score = 0
    for color in [chess.WHITE, chess.BLACK]:
        knights = board.pieces(chess.KNIGHT, color)
        enemy_pawns = board.pieces(chess.PAWN, not color)
        friendly_pawns = board.pieces(chess.PAWN, color)

        for n_sq in knights:
            f = chess.square_file(n_sq)
            r = chess.square_rank(n_sq)

            # 1. Must be in enemy territory
            if color == chess.WHITE and r < 4:
                continue
            if color == chess.BLACK and r > 3:
                continue

            # 2. Must not be attacked by enemy pawn
            attacked = False
            for ep in enemy_pawns:
                er = chess.square_rank(ep)
                ef = chess.square_file(ep)
                if color == chess.WHITE and (er + 1 == r) and abs(ef - f) == 1:
                    attacked = True
                    break
                if color == chess.BLACK and (er - 1 == r) and abs(ef - f) == 1:
                    attacked = True
                    break
            if attacked:
                continue

            # 3. Must be supported by friendly pawn
            supported = False
            for fp in friendly_pawns:
                fr = chess.square_rank(fp)
                ff = chess.square_file(fp)
                if color == chess.WHITE and (fr - 1 == r) and abs(ff - f) == 1:
                    supported = True
                if color == chess.BLACK and (fr + 1 == r) and abs(ff - f) == 1:
                    supported = True

            if not supported:
                continue

            # Award outpost bonus
            score += 20 if color == chess.WHITE else -20

    return score


# Are we in the opening, middlegame, or endgame?

def game_phase(board: chess.Board) -> float:
    # Returns a phase factor between 0 and 1 based on remaining non-pawn material.
    # 1.0 = opening, 0.0 = endgame.

    # Non-pawn pieces considered: Knight, Bishop, Rook, Queen
    max_phase_material = 2 * 320 + 2 * 330 + 2 * 500 + 2 * 900  # total for all non-pawn pieces
    current_material = 0
    for piece_type in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
        current_material += piece_values[piece_type] * (
                    len(board.pieces(piece_type, chess.WHITE)) + len(board.pieces(piece_type, chess.BLACK)))

    phase = current_material / max_phase_material
    return max(0.0, min(1.0, phase))  # Range between (0,1)


# Likely needed for minimax
# def phase(board: chess.Board) -> float:
#    return game_phase(board)


def evaluate_king(board: chess.Board) -> int:
    game_phase_value = game_phase(board)  # 1.0 = opening, 0.0 = endgame
    total_score = 0

    for color in [chess.WHITE, chess.BLACK]:

        # -------------------------
        # Extract king position
        # -------------------------
        king_square = next(iter(board.pieces(chess.KING, color)))
        king_file = chess.square_file(king_square)
        king_rank = chess.square_rank(king_square)

        # -------------------------
        # Middle Game King Safety
        # -------------------------
        king_safety_score = 0

        # King’s home rank (white = 0, black = 7)
        home_rank = 0 if color == chess.WHITE else 7

        # We only count pawn shields when the king is near home
        king_is_near_home = (
            king_rank == home_rank or
            (color == chess.WHITE and king_rank == 1) or
            (color == chess.BLACK and king_rank == 6)
        )

        if king_is_near_home:
            pawn_forward_direction = 1 if color == chess.WHITE else -1

            # Check the three squares directly in front of the king
            for file_offset in [-1, 0, 1]:
                pawn_file = king_file + file_offset
                pawn_rank = king_rank + pawn_forward_direction

                # on board?
                if 0 <= pawn_file <= 7 and 0 <= pawn_rank <= 7:
                    shield_square = chess.square(pawn_file, pawn_rank)
                    piece = board.piece_at(shield_square)

                    if piece and piece.piece_type == chess.PAWN and piece.color == color:
                        king_safety_score += 12  # pawn shield bonus

        # -------------------------
        # Enemy Proximity Penalty
        # -------------------------
        # Only applied in the middlegame (when king should be safe)
        if game_phase_value > 0.35:
            for square, piece in board.piece_map().items():
                if piece.color != color:
                    enemy_file = chess.square_file(square)
                    enemy_rank = chess.square_rank(square)

                    manhattan_distance = abs(king_file - enemy_file) + abs(king_rank - enemy_rank)

                    if manhattan_distance <= 2:
                        king_safety_score -= 3  # small danger penalty

        # -------------------------
        # Endgame: King Activity
        # -------------------------
        # Distance from center (3.5,3.5)
        center_distance = abs(king_file - 3.5) + abs(king_rank - 3.5)
        king_activity_score = -center_distance * 8  # closer to center = better

        # -------------------------
        # Blend based on game phase
        # -------------------------
        king_total = (
            game_phase_value * king_safety_score +
            (1 - game_phase_value) * king_activity_score
        )

        # Add from correct perspective
        if color == chess.WHITE:
            total_score += king_total
        else:
            total_score -= king_total

    return total_score


def evaluate_mobility(board: chess.Board) -> int:
    white_score = 0
    black_score = 0

    # 16-square center (better than just 4 squares)
    central_squares = {
        chess.C3, chess.D3, chess.E3, chess.F3,
        chess.C4, chess.D4, chess.E4, chess.F4,
        chess.C5, chess.D5, chess.E5, chess.F5,
        chess.C6, chess.D6, chess.E6, chess.F6
    }

    opening_phase = 1.0 if board.fullmove_number <= 6 else 0.0

    for move in board.legal_moves:
        piece = board.piece_at(move.from_square)
        if not piece:
            continue

        pts = 0

        # ------------------------------
        # PAWNS
        # ------------------------------
        if piece.piece_type == chess.PAWN:
            pts = 5

            # reward central pawn pushes (very important in small engines)
            if move.to_square in {chess.E4, chess.D4, chess.E5, chess.D5}:
                pts += 25 * opening_phase  # strong push for e4/d4

            # reward moving toward center
            if move.to_square in central_squares:
                pts += 10

        # ------------------------------
        # KNIGHTS
        # ------------------------------
        elif piece.piece_type == chess.KNIGHT:
            pts = 4

            # Opening penalty: knights are OK but not e4-good
            pts -= 6 * opening_phase

        # ------------------------------
        # BISHOPS
        # ------------------------------
        elif piece.piece_type == chess.BISHOP:
            pts = 6

        # ------------------------------
        # ROOKS
        # ------------------------------
        elif piece.piece_type == chess.ROOK:
            pts = 4  # weaker early

            # midgame development bonus
            if not opening_phase:
                pts += 2

        # ------------------------------
        # QUEEN
        # ------------------------------
        elif piece.piece_type == chess.QUEEN:
            pts = 2  # very low early to avoid early queen moves

            # in middlegame queen mobility matters more
            if not opening_phase:
                pts += 6

        # ------------------------------
        # KING (rarely counts)
        # ------------------------------
        elif piece.piece_type == chess.KING:
            pts = 1

        # accumulate score
        if piece.color == chess.WHITE:
            white_score += pts
        else:
            black_score += pts

    return white_score - black_score


def evaluate_center_control(board: chess.Board) -> int:
    score = 0

    # Modern 16-square center
    center_squares = {
        chess.C3, chess.D3, chess.E3, chess.F3,
        chess.C4, chess.D4, chess.E4, chess.F4,
        chess.C5, chess.D5, chess.E5, chess.F5,
        chess.C6, chess.D6, chess.E6, chess.F6,
    }

    # Weight settings (centipawns)
    occupation_values = {
        chess.PAWN: 30,
        chess.KNIGHT: 20,
        chess.BISHOP: 20,
        chess.ROOK: 10,
        chess.QUEEN: 10,
        chess.KING: 0
    }

    control_values = {
        chess.PAWN: 15,
        chess.KNIGHT: 12,
        chess.BISHOP: 10,
        chess.ROOK: 6,
        chess.QUEEN: 4,
        chess.KING: 0
    }

    for sq in center_squares:
        # 1. Occupation bonus
        piece = board.piece_at(sq)
        if piece:
            value = occupation_values.get(piece.piece_type, 0)
            score += value if piece.color == chess.WHITE else -value

        # 2. Control bonus
        attackers_white = board.attackers(chess.WHITE, sq)
        attackers_black = board.attackers(chess.BLACK, sq)

        for attacker in attackers_white:
            p = board.piece_at(attacker)
            score += control_values.get(p.piece_type, 0)

        for attacker in attackers_black:
            p = board.piece_at(attacker)
            score -= control_values.get(p.piece_type, 0)

    return score



# Final calculation function, summation of each score * weight
def evaluate(board: chess.Board) -> int:
    # Game Phase
    phase_value = game_phase(board)

    # weights (not tested, used off of personal experience and intuition)
    w_material = 1.0
    w_pst = (0.3 + 0.7 * phase_value)
    w_pawn_structure = 0.4
    w_bishop_pair = 0.2
    w_knight_outposts = 0.3
    w_rook_files = 0.2
    w_king_safety = 0.6
    w_mobility = 0.3
    w_center = 0.38

    score = 0
    score += w_material * evaluate_material(board)  # material is static
    score += w_pst * evaluate_piece_square_tables(board)  # scale by phase
    score += w_pawn_structure * evaluate_pawn_structure(board)  # static enough
    score += w_bishop_pair * evaluate_bishop_pair(board)  # usually static
    score += w_knight_outposts * evaluate_knight_outposts(board)  # static enough
    score += w_rook_files * evaluate_rook_files(board)  # static enough
    score += w_king_safety * evaluate_king(board)  # king safety changes entirely in endgame
    score += w_mobility * phase_value * evaluate_mobility(board)  # mobility matters more in middlegame
    score += w_center * evaluate_center_control(board)  # At start, focus on moving pawns to the middle

    # Small tempo bonus
    if board.turn == chess.WHITE:
        score += 10
    else:
        score -= 10

    return int(score)

