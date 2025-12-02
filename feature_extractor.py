import chess
import chess.engine
import numpy as np
import feature_extractor_pawns_king as kpfe
import feature_extractor_material as mfe
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
CENTER_OCCUPY_BONUS = np.array([0.08, 0.25, 0.2, 0.12, 0.1, 0.02])
BISHOP_PAIR_WEIGHT = 45
KNIGHT_OUTPOST_WEIGHT = 57
BISHOP_OUTPOST_WEIGHT = 31
PASSED_PAWN_WEIGHT = 10

EARLY_GAME = 0
MID_GAME = 1
END_GAME = 2

feature_weights = {
    "attack_balance": 0,
    "bishop_outposts": BISHOP_OUTPOST_WEIGHT,
    "bishop_pair": BISHOP_PAIR_WEIGHT,
    "bishop_sqr_sum": 0,
    "center_attackers": CENTER_ATTACK_WEIGHTS,
    "connected_pawns": kpfe.CONNECTED_PAWN_WEIGHT,
    "defense_balance": 0,
    "doubled_pawns": kpfe.DOUBLED_PAWN_WEIGHT,
    "half_open_king_files": kpfe.HALF_OPEN_KING_FILES_PEN,
    "isolated_pawns": kpfe.ISOLATED_PAWN_WEIGHT,
    "king_ring_enemy_pressure": kpfe.KING_RING_PRESSURE_WEIGHT,
    "king_sqr_sum": 0,
    "knight_outposts": KNIGHT_OUTPOST_WEIGHT,
    "knight_sqr_sum":0,
    "material_bishop": 0,
    "material_knight": 0,
    "material_pawn": 0,
    "material_queen": 0,
    "material_rook": 0,
    "mobility_balance": 0,
    "mobility_safe_balance": 0,
    "passed_pawns": kpfe.PASSED_PAWN_WEIGHT,
    "pawn_shield": kpfe.KING_SHIELD_WEIGHT,
    "pawn_sqr_sum": 0,
    "pieces_occupying_center": CENTER_OCCUPY_BONUS,
    "queen_sqr_sum": 0,
    "rook_sqr_sum": 0,
    "threat_balance": 0,
}




class FeatureExtractorN:
    def __init__(self, board: chess.Board, game_stage: int):
        self.board = board
        self.feature_count = 40 # TODO: ADJUST TO INCLUDE ALL FEATURES
        self.features = [0 for _ in range(self.feature_count)]
        self.game_stage = game_stage
        self.material_extractor = mfe.MaterialFeatureExtractor(board, game_stage)
        self.pawn_extractor = kpfe.PawnFeatureExtractor(board, game_stage)
        self.king_extractor = kpfe.KingFeatureExtractor(board, game_stage)

    def set_board(self, board: chess.Board):
        self.board = board
        self.material_extractor.set_board(board)
        self.pawn_extractor.set_board(board)
        self.king_extractor.set_board(board)

    # count and subtract number of white vs black pieces occupying the center
    def ft_pieces_occupying_center(self) -> np.array:
        pieces_occupy_center = np.array([0 for _ in range(6)])
        for sq in CENTER:
            if self.board.piece_at(sq):
                if self.board.piece_at(sq).color == chess.WHITE:
                    if self.board.turn == chess.WHITE:
                        pieces_occupy_center[self.board.piece_at(sq).piece_type-1] += 1
                    else:
                        pieces_occupy_center[self.board.piece_at(sq).piece_type-1] -= 1
                elif self.board.piece_at(sq).color == chess.BLACK:
                    if self.board.turn == chess.BLACK:
                        pieces_occupy_center[self.board.piece_at(sq).piece_type-1] += 1
                    else:
                        pieces_occupy_center[self.board.piece_at(sq).piece_type-1] -= 1

        return pieces_occupy_center

    #  count and subtract number of white vs black pieces attacking the center
    def ft_center_attackers(self) -> list:
        attack_counts = np.zeros((2, 6), dtype=np.int32)

        for target in CENTER:
            for sq in self.board.attackers(chess.BLACK, target):
                piece = self.board.piece_at(sq)
                attack_counts[0][piece.piece_type - 1] += 1

            for sq in self.board.attackers(chess.WHITE, target):
                piece = self.board.piece_at(sq)
                attack_counts[1][piece.piece_type - 1] += 1
        if self.board.turn == chess.WHITE:
            return np.subtract(attack_counts[1], attack_counts[0])
        return np.subtract(attack_counts[0], attack_counts[1])

    # check if white and black have both bishops
    def ft_bishop_pair(self) ->int:
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

        print(white_outpost, black_outpost)
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
    def ft_knight_outposts(self)->int:
        return self.outpost(chess.KNIGHT)

    # count and subtract the number of bishop outposts white vs. black
    def ft_bishop_outposts(self) ->int:
        return self.outpost(chess.BISHOP)

    def passed_pawns(self):
        white_pawns = self.board.pieces(chess.PAWN, chess.WHITE)
        white_passed_pawns = 0
        for p in white_pawns:
            if self.is_passed_pawn(p, chess.WHITE):
                white_passed_pawns += 1

        black_pawns = self.board.pieces(chess.PAWN, chess.BLACK)
        black_passed_pawns = 0
        for p in black_pawns:
            if self.is_passed_pawn(p, chess.BLACK):
                black_passed_pawns += 1

        if self.board.turn == chess.WHITE:
            return white_passed_pawns - black_passed_pawns
        return black_passed_pawns - white_passed_pawns

    def is_passed_pawn(self, sqr: chess.Square, color: chess.Color):
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

    def ft_pawn_sqr_sum(self) -> int:
        return self.piece_sqr_sum(chess.PAWN)

    def ft_knight_sqr_sum(self) -> int:
        return self.piece_sqr_sum(chess.KNIGHT)

    def ft_bishop_sqr_sum(self) -> int:
        return self.piece_sqr_sum(chess.BISHOP)

    def ft_rook_sqr_sum(self) -> int:
        return self.piece_sqr_sum(chess.ROOK)

    def ft_queen_sqr_sum(self) -> int:
        return self.piece_sqr_sum(chess.QUEEN)

    def ft_king_sqr_sum(self) -> int:
        if self.game_stage != END_GAME :
            return self.piece_sqr_sum(chess.KING)
        return self.piece_sqr_sum("king_end")

    def ft_passed_pawns(self):
        return self.pawn_extractor.passed_pawns()
    
    def ft_doubled_pawns(self) -> int:
        return self.pawn_extractor.doubled_pawns()
        
    def ft_isolated_pawns(self) -> int:
        return self.pawn_extractor.isolated_pawns()
        
    def ft_connected_pawns(self) -> int:
        return self.pawn_extractor.connected_pawns()
        
    def ft_pawn_shield(self) -> int:
        return self.king_extractor.pawn_shield()
    
    def ft_half_open_king_files(self) -> int:
        return self.king_extractor.half_open_king_files()
        
    def ft_king_ring_enemy_pressure(self) -> int:
        return self.king_extractor.king_ring_enemy_pressure()

    def ft_material_pawn(self):
        return self.material_extractor.ft_material_balance()[0]

    def ft_material_knight(self):
        return self.material_extractor.ft_material_balance()[1]

    def ft_material_bishop(self):
        return self.material_extractor.ft_material_balance()[2]

    def ft_material_rook(self):
        return self.material_extractor.ft_material_balance()[3]

    def ft_material_queen(self):
        return self.material_extractor.ft_material_balance()[4]

    # def ft_material_king(self):
    #     return self.material_extractor.ft_material_balance()[5]

    def ft_mobility_balance(self, safety=False):
        return sum(self.material_extractor.ft_mobility_balance())

    def ft_mobility_safe_balance(self):
        return sum(self.material_extractor.ft_mobility_safe_balance())

    def ft_attack_balance(self):
        return sum(self.material_extractor.ft_attack_balance())

    def ft_threat_balance(self):
        return sum(self.material_extractor.ft_threat_balance())

    def ft_defense_balance(self):
        return sum(self.material_extractor.ft_defense_balance())

    def get_features(self):
        i = 0
        for name, method in inspect.getmembers(self, predicate=inspect.ismethod):
            if not name.startswith("ft_"): continue
            vec = method()
            # print(name[3:], type(vec))
            if name.endswith("pieces_occupying_center") or name.endswith("center_attackers"):
                self.features[i] = np.sum(vec* feature_weights[name[3:]]).item()
            elif isinstance(vec, int):
                if feature_weights[name[3:]] == 0:
                    self.features[i] = vec
                else:
                    self.features[i] = vec * feature_weights[name[3:]]
            i += 1
        return self.features[:i]


def mirror_square(sq: int) -> int:
    return ((7 - (sq // 8)) * 8) + (sq % 8)


def write_features_file(input_file='positions_OLD.csv',output_file='positions_with_features.csv',):
    fe = FeatureExtractorN(chess.Board(), 0)
    feature_names = list(feature_weights.keys())
    feature_names.sort()
    with open(input_file, 'r', encoding='utf-8') as src, open(output_file, 'w', encoding='utf-8', newline='') as dst:
        header = src.readline().rstrip('\n')
        dst.write(header + ',' + ','.join(feature_names) + '\n')

        for line in src:
            line = line.rstrip('\n')
            if not line.strip():
                continue

            parts = line.split(',', 1)
            fen = parts[0]
            rest = parts[1] if len(parts) > 1 else ""

            fe.set_board(chess.Board(fen))
            features = fe.get_features()

            if len(features) != len(feature_names):
                raise ValueError(
                    f"FeatureExtractor returned {len(features)} values, "
                    f"but feature_weights has {len(features)} keys"
                )

            feat_str = ','.join(map(str, features))

            dst.write(f"{fen},{rest},{feat_str}\n")

    print("DONE")


write_features_file()

# fens = [
# "b6b/r6r/q6q/8/8/8/8/8 w KQkq - 0 1",
# # "rnbqkbnr/pppppppp/8/8/8/5N2/PPPPPPPP/RNBQKB1R b KQkq - 1 1",
# # "rnbqkbnr/pppp1ppp/4p3/8/8/5N2/PPPPPPPP/RNBQKB1R w KQkq - 0 2",
# # "rnbqkbnr/pppp1ppp/4p3/8/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 0 2",
# # "rnbqk1nr/ppppbppp/4p3/8/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 1 3",
# ]
#
# fe = mfe.MaterialFeatureExtractor(chess.Board(), 0)
# for i in range(len(fens)):
#     fe.set_board(chess.Board(fens[i]))
#
#     print(fe.get_features())
