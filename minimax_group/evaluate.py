import chess
# NOTE: Weights here are NOT fully fine tuned - minimal testing
# These represent the traditonal centipawn values of the pieces in chess
piece_values = {
    chess.PAWN: 100,
    chess.KNIGHT: 320, # Older models used to use 300 for both Knights/Bishops 
    chess.BISHOP: 330, # but 320/330 split between these is now the common approach
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0, # King does not have an associated centipawn value

}

# Values represent the centipawn bonuses for being on that square -> White's pov based on standard procedures
pawn_table = [
      0,  0,  0,  0,  0,  0,  0,  0,
     50, 50, 50, 50, 50, 50, 50, 50,
     10, 10, 20, 30, 30, 20, 10, 10,
      5,  5, 10, 25, 25, 10,  5,  5,
      0,  0,  0, 20, 20,  0,  0,  0,
      5, -5,-10,  0,  0,-10, -5,  5,
      5, 10, 10,-20,-20, 10, 10,  5,
      0,  0,  0,  0,  0,  0,  0,  0
]

knight_table = [
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50
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

# Combine into one dict
piece_square_tables = {
    chess.PAWN: pawn_table,
    chess.KNIGHT: knight_table,
    chess.BISHOP: bishop_table,
    chess.ROOK: rook_table,
    chess.QUEEN: queen_table,
    chess.KING: king_table
}


def evaluate_material(board: chess.Board) -> int: # This function calculates the material on the board and returns total score
    score = 0
    for square in chess.SQUARES: # Check each square on the board
        piece = board.piece_at(square) 
        if piece: # If there is a piece on that square
            piece_value = piece_values[piece.piece_type]
            if piece.color == chess.WHITE:
                score += piece_value # Add the value of the piece if it belongs to white
            else: # Piece belongs to black
                score -= piece_value # Subtract the value of the piece since not white's

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

def evaluate_pawn_structure(board: chess.Board) -> int:
    score = 0

    for color in [chess.WHITE, chess.BLACK]:
        pawns = list(board.pieces(chess.PAWN, color))
        files = {f: [] for f in range(8)}
        for sq in pawns:
            files[chess.square_file(sq)].append(sq)

        # Doubled pawns
        for f in range(8):
            n = len(files[f])
            if n > 1:
                score += (-15 * (n - 1)) if color == chess.WHITE else (15 * (n - 1))

        # Isolated pawns
        isolated_files = []
        for f in range(8):
            if len(files[f]) == 0:
                continue
            neighbors = []
            if f > 0:
                neighbors.extend(files[f - 1])
            if f < 7:
                neighbors.extend(files[f + 1])
            if len(neighbors) == 0:
                isolated_files.append(f)
                score += (-20) if color == chess.WHITE else 20

        # Passed pawns
        for sq in pawns:
            f = chess.square_file(sq)
            r = chess.square_rank(sq)
            passed = True
            enemy_pawns = board.pieces(chess.PAWN, not color)
            for df in [-1, 0, 1]:
                nf = f + df
                if 0 <= nf <= 7:
                    for ep in enemy_pawns:
                        er = chess.square_rank(ep)
                        ef = chess.square_file(ep)
                        if ef == nf and ((color == chess.WHITE and er > r) or (color == chess.BLACK and er < r)):
                            passed = False
                            break
                    if not passed:
                        break
            if passed:
                bonus = 8 * (r if color == chess.WHITE else 7 - r)
                score += bonus if color == chess.WHITE else -bonus

        # Hanging pawns (two adjacent pawns with no friendly pawns on outside files)
        for i in range(7):
            if i in isolated_files and (i + 1) in isolated_files:
                score -= 15 if color == chess.WHITE else +15

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
        knights = list(board.pieces(chess.KNIGHT, color))
        enemy_pawns = board.pieces(chess.PAWN, not color)
        for n_sq in knights:
            f = chess.square_file(n_sq)
            r = chess.square_rank(n_sq)
            attacked_by_pawn = False
            for ep in enemy_pawns:
                er = chess.square_rank(ep)
                ef = chess.square_file(ep)
                # Pawns attack diagonally
                if color == chess.WHITE and (er + 1 == r) and abs(ef - f) == 1:
                    attacked_by_pawn = True
                    break
                if color == chess.BLACK and (er - 1 == r) and abs(ef - f) == 1:
                    attacked_by_pawn = True
                    break
            if not attacked_by_pawn:
                score += 20 if color == chess.WHITE else -20
    return score

# Are we in the opening, middlegame, or endgame?

def game_phase(board: chess.Board) -> float:
    
    # Returns a phase factor between 0 and 1 based on remaining non-pawn material.
    # 1.0 = opening, 0.0 = endgame.
    
    # Non-pawn pieces considered: Knight, Bishop, Rook, Queen
    max_phase_material = 2*320 + 2*330 + 2*500 + 2*900  # total for all non-pawn pieces
    current_material = 0
    for piece_type in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
        current_material += piece_values[piece_type] * (len(board.pieces(piece_type, chess.WHITE)) + len(board.pieces(piece_type, chess.BLACK)))
    
    phase = current_material / max_phase_material
    return max(0.0, min(1.0, phase))  # Range between (0,1)

# Likely needed for minimax
def phase(board: chess.Board) -> float:
    return game_phase(board)

def evaluate_king(board: chess.Board) -> int:
    phase = game_phase(board)
    score = 0

    for color in [chess.WHITE, chess.BLACK]:
        king_sq = list(board.pieces(chess.KING, color))[0]
        kr, kf = chess.square_rank(king_sq), chess.square_file(king_sq)

        # Middle game
        safety_score = 0
        pawn_dir = 1 if color == chess.WHITE else -1

        # Gradual ramp-up for pawn shield bonus
        ramp_factor = min(board.fullmove_number / 5, 1.0)  # goes from 0 to 1 over first 5 moves
        for df in [-1, 0, 1]:
            r = kr + pawn_dir
            f = kf + df
            if 0 <= r <= 7 and 0 <= f <= 7:
                p = board.piece_at(chess.square(f,r))
                if p and p.piece_type == chess.PAWN and p.color == color:
                    safety_score += 10 * ramp_factor

        # Enemy proximity
        danger = 0
        for sq in chess.SQUARES:
            p = board.piece_at(sq)
            if p and p.color != color:
                pf, pr = chess.square_file(sq), chess.square_rank(sq)
                dist = abs(kf - pf) + abs(kr - pr)
                if dist <= 2:
                    danger += 5 if p.piece_type != chess.PAWN else 2
        safety_score -= danger

        # Endgame activity
        centralization = - (abs(kf - 3.5) + abs(kr - 3.5)) * 10  # Closer to center is better

        king_score = phase * safety_score + (1 - phase) * centralization
        score += king_score if color == chess.WHITE else -king_score

    return score

def evaluate_mobility(board: chess.Board) -> int:
    white_moves = board.legal_moves.count()
    board.push(chess.Move.null())
    black_moves = board.legal_moves.count()
    board.pop()
    return white_moves - black_moves
    
def evaluate_center_control(board):
    score = 0
    center = [chess.E4, chess.D4, chess.E5, chess.D5]

    for sq in center:
        piece = board.piece_at(sq)
        if piece:
            if piece.color == chess.WHITE:
                if piece.piece_type == chess.PAWN:
                    score += 45
                else:
                    score += 15
            else:
                if piece.piece_type == chess.PAWN:
                    score -= 45
                else:
                    score -= 15

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
    w_king_safety = 0.7
    w_mobility = 0.3
    w_center = 0.28

    score = 0
    score += w_material * evaluate_material(board)                              # material is static
    score += w_pst * evaluate_piece_square_tables(board)                        # scale by phase
    score += w_pawn_structure * evaluate_pawn_structure(board)                  # static enough
    score += w_bishop_pair * evaluate_bishop_pair(board)                        # usually static
    score += w_knight_outposts * evaluate_knight_outposts(board)                # static enough
    score += w_rook_files * evaluate_rook_files(board)                          # static enough
    score += w_king_safety * evaluate_king(board)                               # king safety changes entirely in endgame
    score += w_mobility * phase_value * evaluate_mobility(board)                # mobility matters more in middlegame
    score += w_center * evaluate_center_control(board)                          # At start, focus on moving pawns to the middle

    # Small tempo bonus
    if board.turn == chess.WHITE:
        score += 10
    else:
        score -= 10

    return int(score)

