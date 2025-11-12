import chess
import chess.engine
import numpy as np
import inspect

PIECE_SQR_TABLES = {
chess.PAWN :[
    0, 0, 0, 0, 0, 0, 0, 0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
    5, 5, 10, 25, 25, 10, 5, 5,
    0, 0, 0, 20, 20, 0, 0, 0,
    5, -5, -10, 0, 0, -10, -5, 5,
    5, 10, 10, -20, -20, 10, 10, 5,
    0, 0, 0, 0, 0, 0, 0, 0
],
chess.KNIGHT: [
    -50, -40, -30, -30, -30, -30, -40, -50,
    -40, -20, 0, 0, 0, 0, -20, -40,
    -30, 0, 10, 15, 15, 10, 0, -30,
    -30, 5, 15, 20, 20, 15, 5, -30,
    -30, 0, 15, 20, 20, 15, 0, -30,
    -30, 5, 10, 15, 15, 10, 5, -30,
    -40, -20, 0, 5, 5, 0, -20, -40,
    -50, -40, -30, -30, -30, -30, -40, -50,
],chess.BISHOP:[
    -20, -10, -10, -10, -10, -10, -10, -20,
    -10, 0, 0, 0, 0, 0, 0, -10,
    -10, 0, 5, 10, 10, 5, 0, -10,
    -10, 5, 5, 10, 10, 5, 5, -10,
    -10, 0, 10, 10, 10, 10, 0, -10,
    -10, 10, 10, 10, 10, 10, 10, -10,
    -10, 5, 0, 0, 0, 0, 5, -10,
    -20, -10, -10, -10, -10, -10, -10, -20,
], chess.ROOK:[
    0, 0, 0, 0, 0, 0, 0, 0,
    5, 10, 10, 10, 10, 10, 10, 5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    0, 0, 0, 5, 5, 0, 0, 0
], chess.QUEEN:[
    -20, -10, -10, -5, -5, -10, -10, -20,
    -10, 0, 0, 0, 0, 0, 0, -10,
    -10, 0, 5, 5, 5, 5, 0, -10,
    -5, 0, 5, 5, 5, 5, 0, -5,
    0, 0, 5, 5, 5, 5, 0, -5,
    -10, 5, 5, 5, 5, 5, 0, -10,
    -10, 0, 5, 0, 0, 0, 0, -10,
    -20, -10, -10, -5, -5, -10, -10, -20
],chess.KING:[
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -20, -30, -30, -40, -40, -30, -30, -20,
    -10, -20, -20, -20, -20, -20, -20, -10,
    20, 20, 0, 0, 0, 0, 20, 20,
    20, 30, 10, 0, 0, 10, 30, 20
], "king_end":[
    -50, -40, -30, -20, -20, -30, -40, -50,
    -30, -20, -10, 0, 0, -10, -20, -30,
    -30, -10, 20, 30, 30, 20, -10, -30,
    -30, -10, 30, 40, 40, 30, -10, -30,
    -30, -10, 30, 40, 40, 30, -10, -30,
    -30, -10, 20, 30, 30, 20, -10, -30,
    -30, -30, 0, 0, 0, 0, -30, -30,
    -50, -30, -30, -30, -30, -30, -30, -50
]}

CENTER = {chess.D4, chess.E4, chess.D5, chess.E5}

CENTER_ATTACK_WEIGHTS = np.array([5,  15,  12, 10, 8, 3]) # [pawn, knight, bishop, rook, queen, king]
CENTER_OCCUPY_BONUS = np.array([0.08, 0,25, 0.2, 0.12, 0.1, 0.02])

EARLY_GAME = 0
MID_GAME = 1
END_GAME = 2

# weights
PASSED_PAWN_WEIGHT = 10
DOUBLED_PAWN_WEIGHT = -20
ISOLATED_PAWN_WEIGHT = -20
CONNECTED_PAWN_WEIGHT = 8

KING_SHIELD_WEIGHT = 10
HALF_OPEN_KING_FILES_PEN = -12
KING_RING_PRESSURE_WEIGHT = -3
BISHOP_PAIR_WEIGHT = 45
KNIGHT_OUTPOST_WEIGHT = 57
BISHOP_OUTPOST_WEIGHT = 31

# Attacker weights for ring pressure
ATTACKER_WEIGHT = {
    chess.PAWN: 1,
    chess.KNIGHT: 2,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 5,
    chess.KING: 0,
}

PIECE_VALUE = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,
}

def mirror_square(sq: int) -> int:
    return ((7 - (sq // 8)) * 8) + (sq % 8)

def determine_stage(move_num: int) -> int:
    if move_num <= 15:
        return EARLY_GAME
    if move_num <= 35:
        return MID_GAME
    return END_GAME

class FeatureExtractor:
    def __init__(self, board: chess.Board, game_stage: int):
        self.board = board
        self.feature_count = 12 + 9 + 65 # TODO: ADJUST TO INCLUDE ALL FEATURES
        self.features = [0] * self.feature_count
        self.game_stage = game_stage

    def get_features(self):
        return self.features

    # count and subtract number of white vs black pieces occupying the center
    def pieces_occupying_center(self) -> np.array:
        pieces_occupy_center = np.array([0 for _ in range(6)])

        for sq in CENTER:
            if self.board.piece_at(sq).color == chess.WHITE:
                if self.board.turn == chess.WHITE:
                    pieces_occupy_center[self.board.piece_at(sq).piece_type] += 1
                else:
                    pieces_occupy_center[self.board.piece_at(sq).piece_type] -= 1
            elif self.board.piece_at(sq).color == chess.BLACK:
                if self.board.turn == chess.BLACK:
                    pieces_occupy_center[self.board.piece_at(sq).piece_type] += 1
                else:
                    pieces_occupy_center[self.board.piece_at(sq).piece_type] -= 1
        return pieces_occupy_center

    #  count and subtract number of white vs black pieces attacking the center
    def center_attackers(self) -> list:
        attack_counts = np.zeros((2, 6), dtype=np.int32)
        for target in CENTER:
            for sq in self.board.attackers(chess.BLACK, target):
                piece = self.board.piece_at(sq)
                attack_counts[chess.BLACK][piece.piece_type - 1] += 1

            for sq in self.board.attackers(chess.WHITE, target):
                piece = self.board.piece_at(sq)
                attack_counts[chess.WHITE][piece.piece_type - 1] += 1

        if self.board.turn == chess.WHITE:
            return np.subtract(attack_counts[1], attack_counts[0])
        return np.subtract(attack_counts[0], attack_counts[1])

    # check if white and black have both bishops
    def bishop_pair(self) ->int:
        white_has_bp = 1 if len(self.board.pieces(chess.BISHOP, chess.WHITE)) == 2 else 0
        black_has_bp = 1 if len(self.board.pieces(chess.BISHOP, chess.BLACK)) == 2 else 0
        if self.board.turn == chess.WHITE:
            return white_has_bp - black_has_bp
        return black_has_bp - white_has_bp

    def outpost(self, piece_type: chess.PieceType)->int:
        white_outpost = 0
        black_outpost = 0
        # find white outposts
        for sqr in self.board.pieces(piece_type, chess.WHITE):
            if self.is_outpost(sqr, chess.WHITE, piece_type):
                white_outpost += 1

        # find black outposts
        for sqr in self.board.pieces(piece_type, chess.BLACK):
            if self.is_outpost(sqr, chess.BLACK, piece_type):
                black_outpost += 1

        if self.board.turn == chess.WHITE:
            return white_outpost - black_outpost
        return black_outpost - white_outpost

    # determine whether the square is an outpost
    def is_outpost(self, sqr: chess.Square, color: chess.Color, piece_type: chess.PieceType)->bool:
        piece = self.board.piece_at(sqr)
        # check if input is valid
        if not piece or piece.color != color or piece.piece_type != piece_type:
            return False

        opp_color = not color
        # is supported by pawns
        if chess.PAWN not in [self.board.piece_at(sqr).piece_type for sqr in self.board.attackers(color, sqr)]:
            return False

        # is not attacked by opposite pawns
        if chess.PAWN in [self.board.piece_at(sqr).piece_type for sqr in self.board.attackers(opp_color, sqr)]:
            return False

        # is on opponents half
        rank = chess.square_rank(sqr)
        if color == chess.WHITE and rank < 4:
            return False
        elif color == chess.BLACK and rank > 3:
            return False

        return True

    # count and subtract the number of knight outposts white vs. black
    def knight_outposts(self)->int:
        return self.outpost(chess.KNIGHT)

    # count and subtract the number of bishop outposts white vs. black
    def bishop_outposts(self) ->int:
        return self.outpost(chess.BISHOP)

    def passed_pawns_1(self):
        white_pawns = self.board.pieces(chess.PAWN, chess.WHITE)
        white_passed_pawns = 0
        for p in white_pawns:
            if self.is_passed_pawn_1(p, chess.WHITE):
                white_passed_pawns += 1

        black_pawns = self.board.pieces(chess.PAWN, chess.BLACK)
        black_passed_pawns = 0
        for p in black_pawns:
            if self.is_passed_pawn_1(p, chess.BLACK):
                black_passed_pawns += 1

        if self.board.turn == chess.WHITE:
            return white_passed_pawns - black_passed_pawns
        return black_passed_pawns - white_passed_pawns

    def is_passed_pawn_1(self, sqr: chess.Square, color: chess.Color):
        enemy_pawns = self.board.pieces(chess.PAWN, not color)

        file_index = chess.square_file(sqr)
        rank_index = chess.square_rank(sqr)

        for p in enemy_pawns:
            if color == chess.WHITE:
                if chess.square_file(p) in [file_index-1, file_index, file_index+1] and chess.square_rank(p) > rank_index:
                    return False
            else:
                if chess.square_file(p) in [file_index-1, file_index, file_index+1] and chess.square_rank(p) < rank_index:
                    return False
        return True

    def piece_sqr_sum_color(self, color: chess.Color, piece_type: chess.PieceType, table_name):
        if table_name not in PIECE_SQR_TABLES:
            return 0
        sqr_sum = 0
        piece_sqrs = self.board.pieces(piece_type, color)

        if len(piece_sqrs) == 0:
            return 0

        for sqr in piece_sqrs:
            if color == chess.WHITE:
                sqr_sum += PIECE_SQR_TABLES[table_name][sqr]
            else:
                sqr_sum += PIECE_SQR_TABLES[table_name][mirror_square(sqr)]
        return sqr_sum

    def piece_sqr_sum(self, piece_type) -> int:
        white_piece_sqr_sum = self.piece_sqr_sum_color(chess.WHITE, piece_type, piece_type)
        black_piece_sqr_sum =self.piece_sqr_sum_color(chess.BLACK, piece_type, piece_type)
        if self.board.turn == chess.WHITE:
            return white_piece_sqr_sum - black_piece_sqr_sum
        return black_piece_sqr_sum - white_piece_sqr_sum

    def pawn_square_sum(self) -> int:
        return self.piece_sqr_sum(chess.PAWN)

    def knight_sqr_sum(self) -> int:
        return self.piece_sqr_sum(chess.KNIGHT)

    def bishop_sqr_sum(self) -> int:
        return self.piece_sqr_sum(chess.BISHOP)

    def rook_sqr_sum(self) -> int:
        return self.piece_sqr_sum(chess.ROOK)

    def queen_sqr_sum(self) -> int:
        return self.piece_sqr_sum(chess.QUEEN)

    def king_sqr_sum(self) -> int:
        if self.game_stage != END_GAME :
            return self.piece_sqr_sum(chess.KING)
        return self.piece_sqr_sum("king_end")

    def ft_compute_1(self):
        features = [0] * 12
        features[0] = (self.pieces_occupying_center() * CENTER_OCCUPY_BONUS).sum()
        features[1] = (self.center_attackers() * CENTER_ATTACK_WEIGHTS).sum()
        features[2] = self.bishop_pair()*BISHOP_PAIR_WEIGHT
        features[3] = self.knight_outposts()*KNIGHT_OUTPOST_WEIGHT
        features[4] = self.bishop_outposts()*BISHOP_OUTPOST_WEIGHT
        features[5] = self.pawn_square_sum()
        features[6] = self.knight_sqr_sum()
        features[7] = self.bishop_sqr_sum()
        features[8] = self.rook_sqr_sum()
        features[9] = self.queen_sqr_sum()
        features[10] = self.king_sqr_sum()
        features[11] = self.passed_pawns_1()*PASSED_PAWN_WEIGHT
        return features

    ##################################################################################
    ##################################################################################
    ##################################################################################

    def passed_pawns_2(self):
        white_pawns = self.board.pieces(chess.PAWN, chess.WHITE)
        white_passed_pawns = 0
        for p in chess.SquareSet(white_pawns):
            if self.is_passed_pawn_2(p, chess.WHITE):
                white_passed_pawns += 1

        black_pawns = self.board.pieces(chess.PAWN, chess.BLACK)
        black_passed_pawns = 0
        for p in chess.SquareSet(black_pawns):
            if self.is_passed_pawn_2(p, chess.BLACK):
                black_passed_pawns += 1

        if self.board.turn == chess.WHITE:
            return white_passed_pawns - black_passed_pawns
        return black_passed_pawns - white_passed_pawns

    def is_passed_pawn_2(self, sqr: chess.Square, color: chess.Color):
        enemy_pawns = self.board.pieces(chess.PAWN, not color)
        file_index = chess.square_file(sqr)
        rank_index = chess.square_rank(sqr)

        for p in chess.SquareSet(enemy_pawns):
            pf = chess.square_file(p)
            pr = chess.square_rank(p)
            if color == chess.WHITE:
                if pf in (file_index-1, file_index, file_index+1) and pr > rank_index:
                    return False
            else:
                if pf in (file_index-1, file_index, file_index+1) and pr < rank_index:
                    return False
        return True

    def doubled_pawns(self) -> int:
        def count(color: chess.Color) -> int:
            pawns = self.board.pieces(chess.PAWN, color)
            total = 0
            for f in range(8):
                n = len(pawns & chess.BB_FILES[f])
                if n > 1:
                    total += (n - 1)
            return total
        w, b = count(chess.WHITE), count(chess.BLACK)
        if self.board.turn == chess.WHITE:
            return (w - b)
        return (b - w)

    def isolated_pawns(self) -> int:
        def count(color: chess.Color) -> int:
            pawns = self.board.pieces(chess.PAWN, color)
            total = 0
            for p in chess.SquareSet(pawns):
                f = chess.square_file(p)
                left_mask  = chess.BB_FILES[f-1] if f > 0 else 0
                right_mask = chess.BB_FILES[f+1] if f < 7 else 0
                if (pawns & (left_mask | right_mask)) == 0:
                    total += 1
            return total

        w = count(chess.WHITE)
        b = count(chess.BLACK)
        if self.board.turn == chess.WHITE:
            return (w - b)
        return (b - w)

    def connected_pawns(self) -> int:
        def count(color: chess.Color) -> int:
            pawns = self.board.pieces(chess.PAWN, color)
            total = 0

            # phalanx: adjacent files, same rank
            for f in range(7):
                pf   = pawns & chess.BB_FILES[f]
                pfp1 = pawns & chess.BB_FILES[f+1]
                if pf and pfp1:
                    for sq in chess.SquareSet(pf):
                        r = chess.square_rank(sq)
                        if pfp1 & chess.BB_RANKS[r]:
                            total += 1

            # pawn chain: defended from behind-diagonal
            dir_ = -1 if color == chess.WHITE else 1
            for p in chess.SquareSet(pawns):
                f = chess.square_file(p)
                r = chess.square_rank(p)
                sr = r + dir_
                if 0 <= sr <= 7:
                    left_connected  = (f > 0) and ((pawns >> chess.square(f-1, sr)) & 1)
                    right_connected = (f < 7) and ((pawns >> chess.square(f+1, sr)) & 1)
                    if left_connected or right_connected:
                        total += 1
            return total

        w = count(chess.WHITE)
        b = count(chess.BLACK)
        if self.board.turn == chess.WHITE:
            return (w - b)
        return (b - w)

    def compute_2(self) -> dict:
        passed = self.passed_pawns_2() * PASSED_PAWN_WEIGHT
        doubled = self.doubled_pawns() * DOUBLED_PAWN_WEIGHT
        isolated = self.isolated_pawns() * ISOLATED_PAWN_WEIGHT
        connected = self.connected_pawns() * CONNECTED_PAWN_WEIGHT

        pawn_total = passed + doubled + isolated + connected
        return {
            "passed": passed,
            "doubled": doubled,
            "isolated": isolated,
            "connected": connected,
            "pawn_total": pawn_total,
        }

    def ft_compute_2(self):
        return list(self.compute_2().values())

    def pawn_shield(self) -> int:
        def shield(color: chess.Color) -> int:
            ksq = self.board.king(color)
            if ksq is None:
                return 0
            pawns = self.board.pieces(chess.PAWN, color)
            kf = chess.square_file(ksq)
            kr = chess.square_rank(ksq)
            dir_ = 1 if color == chess.WHITE else -1
            s = 0
            for df in (-1, 0, 1):
                f = kf + df
                if 0 <= f <= 7:
                    r1 = kr + dir_
                    r2 = kr + 2*dir_
                    if 0 <= r1 <= 7 and ((pawns >> chess.square(f, r1)) & 1):
                        s += 2
                    if 0 <= r2 <= 7 and ((pawns >> chess.square(f, r2)) & 1):
                        s += 1
            return s

        w = shield(chess.WHITE)
        b = shield(chess.BLACK)
        if self.board.turn == chess.WHITE:
            return w - b
        return b - w

    def half_open_king_files(self) -> int:
        def half_open(color: chess.Color) -> int:
            ksq = self.board.king(color)
            if ksq is None:
                return 0
            pawns = self.board.pieces(chess.PAWN, color)
            kf = chess.square_file(ksq)
            total = 0
            for df in (-1, 0, 1):
                f = kf + df
                if 0 <= f <= 7 and (pawns & chess.BB_FILES[f]) == 0:
                    total += 1
            return total

        w = half_open(chess.WHITE)
        b = half_open(chess.BLACK)
        if self.board.turn == chess.WHITE:
            return w - b
        return b - w

    def king_ring_enemy_pressure(self) -> int:
        def pressure(color: chess.Color) -> int:
            ksq = self.board.king(color)
            if ksq is None:
                return 0
            enemy = not color
            ring = chess.BB_KING_ATTACKS[ksq] | chess.BB_SQUARES[ksq]
            total = 0
            for sq in chess.SquareSet(ring):
                for a in self.board.attackers(enemy, sq):
                    piece = self.board.piece_at(a)
                    if piece:
                        total += ATTACKER_WEIGHT.get(piece.piece_type, 0)
            return total

        w = pressure(chess.WHITE)
        b = pressure(chess.BLACK)
        if self.board.turn == chess.WHITE:
            return w - b
        return b - w

    def compute_3(self) -> dict:
        shield = self.pawn_shield() * KING_SHIELD_WEIGHT
        halfopen = self.half_open_king_files() * HALF_OPEN_KING_FILES_PEN
        press = self.king_ring_enemy_pressure() * KING_RING_PRESSURE_WEIGHT
        king_total = shield + halfopen + press
        return {
            "king_shield": shield,
            "king_half_open": halfopen,
            "king_pressure": press,
            "king_total": king_total,
        }

    def ft_compute_3(self):
        return list(self.compute_3().values())

    ##################################################################################
    ##################################################################################
    ##################################################################################

    def ft_material_balance(self):
        # difference in total material values for each piece type between the players
        board = self.board
        scores = []
        side = 1
        if board.turn == chess.BLACK: side = -1
        for piece_type, v in PIECE_VALUE.items():
            if piece_type != chess.KING:
                scores.append(side * v * (len(board.pieces(piece_type, chess.WHITE)) - len(board.pieces(piece_type, chess.BLACK))))
        return scores

    def ft_mobility_one_side(self, safety = False):
        # total number of legal squares each piece type can move to for the current player
        board = self.board
        scores = [0] * 6
        attacked_squares = set()
        if safety:
            for piece_type in chess.PIECE_TYPES:
                for square in board.pieces(piece_type, not board.turn):
                    attacked_squares.update(board.attacks(square))
        for move in list(board.legal_moves):
            if safety and move.to_square in attacked_squares: continue
            piece = board.piece_at(move.from_square)
            if piece: scores[piece.piece_type - 1] += 1
        return scores

    def ft_mobility_balance(self, safety = False):
        # difference in total number of legal squares each piece type can move to between the players
        board = self.board
        scores = self.ft_mobility_one_side(safety)
        board.push(chess.Move.null())
        scores_other = self.ft_mobility_one_side(safety)
        board.pop()
        for i in range(len(scores)):
            scores[i] -= scores_other[i]
        return scores

    def ft_mobility_safe_one_side(self):
        # total number of safe legal squares each piece type can move to for the current player
        return self.ft_mobility_one_side(True)

    def ft_mobility_safe_balance(self):
        # difference in total number of safe legal squares each piece type can move to between the players
        return self.ft_mobility_balance(True)

    def ft_attack_one_side(self):
        # total number of squares attacked by each piece type for the current player
        board = self.board
        scores = [0] * 6
        for piece_type in chess.PIECE_TYPES:
            for square in board.pieces(piece_type, board.turn):
                scores[piece_type - 1] += len(board.attacks(square))
        return scores

    def ft_attack_balance(self):
        # difference in total number of squares attacked by each piece type between the players
        board = self.board
        scores = self.ft_attack_one_side()
        board.push(chess.Move.null())
        scores_other = self.ft_attack_one_side()
        board.pop()
        for i in range(len(scores)):
            scores[i] -= scores_other[i]
        return scores

    def ft_threat_one_side(self):
        # total number of opponent pieces threatened by each piece type for the current player
        board = self.board
        scores = [0] * 6
        for piece_type in chess.PIECE_TYPES:
            for square in board.pieces(piece_type, board.turn):
                for attacked_square in board.attacks(square):
                    attacked_piece = board.piece_at(attacked_square)
                    if attacked_piece and attacked_piece.color != board.turn:
                        scores[piece_type - 1] += 1
        return scores

    def ft_threat_balance(self):
        # difference in threats to enemy pieces by each piece type between the players
        board = self.board
        scores = self.ft_threat_one_side()
        board.push(chess.Move.null())
        scores_other = self.ft_threat_one_side()
        board.pop()
        for i in range(len(scores)):
            scores[i] -= scores_other[i]
        return scores

    def ft_defense_one_side(self):
        # total number of friendly pieces defending each piece type for the current player
        board = self.board
        scores = [0] * 6
        for piece_type in chess.PIECE_TYPES:
            for square in board.pieces(piece_type, board.turn):
                defenders = board.attackers(board.turn, square)
                if defenders: scores[piece_type - 1] += len(defenders)
        return scores

    def ft_defense_balance(self):
        # difference in total number of defenders for each piece type between the players
        board = self.board
        scores = self.ft_defense_one_side()
        board.push(chess.Move.null())
        scores_other = self.ft_defense_one_side()
        board.pop()
        for i in range(len(scores)):
            scores[i] -= scores_other[i]
        return scores

    def extract_features(self):
        i = 0
        for name, method in sorted(inspect.getmembers(self, predicate=inspect.ismethod)):
            if not name.startswith("ft_"): continue
            vec = method()
            for value in vec:
                self.features[i] = value
                i += 1
        return self.features

fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
fe = FeatureExtractor(chess.Board(fen), 0)
print(fe.get_features())

fen2 = "r1b1k1nr/p2p1ppp/2nPpb2/qpp1P3/8/5N2/PPP1QPPP/RNB1KB1R w KQkq - 1 8"
fe.set_board(chess.Board(fen2))
print(fe.get_features())

fen3 = "6k1/p4p2/2pN3p/8/r7/7P/5PPK/8 w - - 4 38"
fe.set_board(chess.Board(fen3))
print(fe.get_features())