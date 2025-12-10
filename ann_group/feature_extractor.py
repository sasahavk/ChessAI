import chess
import chess.engine
import numpy as np
import ann_group.feature_extractor_pawns_king as kpfe
import ann_group.feature_extractor_material as mfe
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
    "attack_black": 0,
    "attack_white": 0,

    "bishop_outposts_black": 0,
    "bishop_outposts_white": 0,

    "bishop_pair_white": 0,
    "bishop_pair_black": 0,

    "bishop_sqr_sum": 0,
    "bishop_sqr_sum_black": 0,
    "bishop_sqr_sum_white": 0,

    # "bknrk_sqr_sum": 0,

    "center_attackers_white": CENTER_ATTACK_WEIGHTS,
    "center_attackers_black": CENTER_ATTACK_WEIGHTS,

    "connected_pawns": 0,
    "connected_pawns_black": 0,
    "connected_pawns_white": 0,

    "defense_balance": 0,
    "defense_black": 0,
    "defense_white": 0,

    "doubled_pawns": 0,
    "doubled_pawns_black": 0,
    "doubled_pawns_white": 0,

    "half_open_king_files": 0,
    "half_open_king_files_black": 0,
    "half_open_king_files_white": 0,

    "isolated_pawns": 0,
    "isolated_pawns_black": 0,
    "isolated_pawns_white": 0,

    "king_ring_enemy_pressure": 0,
    "king_ring_enemy_pressure_black": 0,
    "king_ring_enemy_pressure_white": 0,

    "king_sqr_sum": 0,
    "king_sqr_sum_black": 0,
    "king_sqr_sum_white": 0,

    "knight_outposts_black": 0,
    "knight_outposts_white": 0,

    "knight_sqr_sum":0,
    "knight_sqr_sum_black": 0,
    "knight_sqr_sum_white": 0,

    "material_bishop_white": 0,
    "material_bishop_black": 0,

    "material_knight_white": 0,
    "material_knight_black": 0,

    "material_pawn_white": 0,
    "material_pawn_black": 0,

    "material_queen_white": 0,
    "material_queen_black": 0,

    "material_rook_white": 0,
    "material_rook_black": 0,

    "mobility_balance": 0,
    "mobility_black": 0,
    "mobility_white": 0,

    "mobility_safe_balance": 0,
    "mobility_safe_black": 0,
    "mobility_safe_white": 0,

    "outposts_white":0,
    "outposts_black": 0,

    "passed_pawns": 0,
    "passed_pawns_black": 0,
    "passed_pawns_white": 0,

    "pawn_shield": 0,
    "pawn_shield_black": 0,
    "pawn_shield_white": 0,

    "pawn_sqr_sum": 0,
    "pawn_sqr_sum_black": 0,
    "pawn_sqr_sum_white": 0,

    "pieces_occupying_center": CENTER_OCCUPY_BONUS,

    "queen_sqr_sum": 0,
    "queen_sqr_sum_black": 0,
    "queen_sqr_sum_white": 0,

    "rook_sqr_sum": 0,
    "rook_sqr_sum_black": 0,
    "rook_sqr_sum_white": 0,

    "sqr_sum_black":0,
    "sqr_sum_white":0,
    "target": 0,

    "threat_balance": 0,
    "threat_black": 0,
    "threat_white": 0,
}


class FeatureExtractorN:
    def __init__(self, board: chess.Board, game_stage: int, stockfish_path:str):
        self.board = board
        self.feature_count = 50
        self.features = [0 for _ in range(self.feature_count)]
        # features used in ANN prediction
        self.feature_functions = {
            "attack_black": self.ft_attack_black,
            "attack_white": self.ft_attack_white,
            "bishop_pair_white": self.ft_bishop_pair_white,
            "bishop_pair_black": self.ft_bishop_pair_black,
            "center_attackers_white": self.ft_center_attackers_white,
            "center_attackers_black": self.ft_center_attackers_black,
            "connected_pawns": self.ft_connected_pawns,
            "defense_balance": self.ft_defense_balance,
            "doubled_pawns_black": self.ft_doubled_pawns_black,
            "doubled_pawns_white": self.ft_doubled_pawns_white,
            "half_open_king_files": self.ft_half_open_king_files,
            "isolated_pawns_black": self.ft_isolated_pawns_black,
            "isolated_pawns_white": self.ft_isolated_pawns_white,
            "king_ring_enemy_pressure": self.ft_king_ring_enemy_pressure,
            "material_bishop_white": self.ft_material_bishop_white,
            "material_bishop_black": self.ft_material_bishop_black,
            "material_knight_white": self.ft_material_knight_white,
            "material_knight_black": self.ft_material_knight_black,
            "material_pawn_white": self.ft_material_pawn_white,
            "material_pawn_black": self.ft_material_pawn_black,
            "material_queen_white": self.ft_material_queen_white,
            "material_queen_black": self.ft_material_queen_black,
            "material_rook_white": self.ft_material_rook_white,
            "material_rook_black": self.ft_material_rook_black,
            "mobility_black": self.ft_mobility_black,
            "mobility_white": self.ft_mobility_white,
            "mobility_safe_black": self.ft_mobility_safe_black,
            "mobility_safe_white": self.ft_mobility_safe_white,
            "outposts_white": self.ft_outposts_white,
            "outposts_black": self.ft_outposts_black,
            "passed_pawns": self.ft_passed_pawns,
            "pawn_shield": self.ft_pawn_shield,
            "pieces_occupying_center": self.ft_pieces_occupying_center,
            "sqr_sum_black": self.ft_sqr_sum_black,
            "sqr_sum_white": self.ft_sqr_sum_white,
            "threat_black": self.ft_threat_black,
            "threat_white": self.ft_threat_white,
        }
        self.game_stage = game_stage
        self.material_extractor = mfe.MaterialFeatureExtractor(board, game_stage)
        self.pawn_extractor = kpfe.PawnFeatureExtractor(board, game_stage)
        self.king_extractor = kpfe.KingFeatureExtractor(board, game_stage)
        self.engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)

    def set_board(self, board: chess.Board):
        self.board = board
        self.material_extractor.set_board(board)
        self.pawn_extractor.set_board(board)
        self.king_extractor.set_board(board)

    def ft_attack_balance(self):
        if self.board.turn == chess.WHITE:
            return sum(self.material_extractor.ft_attack_balance())
        return -sum(self.material_extractor.ft_attack_balance())

    def ft_attack_white(self):
        if self.board.turn == chess.WHITE:
            return sum(self.material_extractor.attack(chess.WHITE))
        return -sum(self.material_extractor.attack(chess.WHITE))

    def ft_attack_black(self):
        if self.board.turn == chess.BLACK:
            return sum(self.material_extractor.attack(chess.BLACK))
        return -sum(self.material_extractor.attack(chess.BLACK))


    def ft_bishop_outposts_white(self):
        if self.board.turn == chess.WHITE:
            return self.bishop_outposts(chess.WHITE)
        return -(self.bishop_outposts(chess.WHITE))

    def ft_bishop_outposts_black(self):
        if self.board.turn == chess.BLACK:
            return self.bishop_outposts(chess.BLACK)
        return -(self.bishop_outposts(chess.BLACK))


    # check if white and black have both bishops
    def bishop_pair(self, color:chess.Color) ->int:
        return 1 if len(self.board.pieces(chess.BISHOP, color)) == 2 else 0

    def ft_bishop_pair_white(self):
        if self.board.turn == chess.WHITE:
            return self.bishop_pair(chess.WHITE)*BISHOP_PAIR_WEIGHT
        return -(self.bishop_pair(chess.WHITE))*BISHOP_PAIR_WEIGHT

    def ft_bishop_pair_black(self):
        if self.board.turn == chess.BLACK:
            return self.bishop_pair(chess.BLACK)*BISHOP_PAIR_WEIGHT
        return -(self.bishop_pair(chess.BLACK) )*BISHOP_PAIR_WEIGHT


    def ft_bishop_sqr_sum(self) -> int:
        return self.piece_sqr_sum(chess.BISHOP)

    def ft_bishop_sqr_sum_white(self):
        if self.board.turn == chess.WHITE:
            return  self.piece_sqr_sum_color(chess.WHITE, chess.BISHOP, chess.BISHOP)
        return -self.piece_sqr_sum_color(chess.WHITE, chess.BISHOP, chess.BISHOP)

    def ft_bishop_sqr_sum_black(self):
        if self.board.turn == chess.BLACK:
            return  self.piece_sqr_sum_color(chess.BLACK, chess.BISHOP, chess.BISHOP)
        return -self.piece_sqr_sum_color(chess.BLACK, chess.BISHOP, chess.BISHOP)


    #  count and subtract number of white vs black pieces attacking the center
    def center_attackers(self, color: chess.Color) :
        attack_counts = np.zeros((1, 6), dtype=np.int32)

        for target in CENTER:
            for sq in self.board.attackers(color, target):
                piece = self.board.piece_at(sq)
                attack_counts[0][piece.piece_type - 1] += 1

        return  np.sum(attack_counts * CENTER_ATTACK_WEIGHTS).item()

    def ft_center_attackers_white(self):
        if self.board.turn == chess.WHITE:
            return self.center_attackers(chess.WHITE)
        return -self.center_attackers(chess.WHITE)

    def ft_center_attackers_black(self):
        if self.board.turn == chess.BLACK:
            return self.center_attackers(chess.BLACK)
        return -self.center_attackers(chess.BLACK)


    def ft_connected_pawns(self) -> int:
        return self.pawn_extractor.connected_pawns()

    def ft_connected_pawns_white(self):
        if self.board.turn == chess.WHITE:
            return self.pawn_extractor.connected_pawns_color(chess.WHITE)
        return -self.pawn_extractor.connected_pawns_color(chess.WHITE)

    def ft_connected_pawns_black(self):
        if self.board.turn == chess.BLACK:
            return self.pawn_extractor.connected_pawns_color(chess.BLACK)
        return -self.pawn_extractor.connected_pawns_color(chess.BLACK)


    def ft_defense_balance(self):
        if self.board.turn == chess.WHITE:
            return sum(self.material_extractor.ft_defense_balance())
        return -sum(self.material_extractor.ft_defense_balance())

    def ft_defense_white(self):
        if self.board.turn == chess.WHITE:
            return sum(self.material_extractor.defense(chess.WHITE))
        return -sum(self.material_extractor.defense(chess.WHITE))

    def ft_defense_black(self):
        if self.board.turn == chess.BLACK:
            return sum(self.material_extractor.defense(chess.BLACK))
        return -sum(self.material_extractor.defense(chess.BLACK))


    def ft_doubled_pawns(self) -> int:
        return self.pawn_extractor.doubled_pawns()

    def ft_doubled_pawns_white(self):
        if self.board.turn == chess.WHITE:
            return  self.pawn_extractor.doubled_pawns_color(chess.WHITE)
        return -self.pawn_extractor.doubled_pawns_color(chess.WHITE)

    def ft_doubled_pawns_black(self):
        if self.board.turn == chess.BLACK:
            return self.pawn_extractor.doubled_pawns_color(chess.BLACK)
        return -self.pawn_extractor.doubled_pawns_color(chess.BLACK)


    def ft_half_open_king_files(self) -> int:
        return self.king_extractor.half_open_king_files()

    def ft_half_open_king_files_white(self) -> int:
        if self.board.turn == chess.WHITE:
            return self.king_extractor.half_open_king_files_color(chess.WHITE)
        return -self.king_extractor.half_open_king_files_color(chess.WHITE)

    def ft_half_open_king_files_black(self) -> int:
        if self.board.turn == chess.BLACK:
            return self.king_extractor.half_open_king_files_color(chess.BLACK)
        return -self.king_extractor.half_open_king_files_color(chess.BLACK)


    def ft_isolated_pawns(self) -> int:
        return self.pawn_extractor.isolated_pawns()

    def ft_isolated_pawns_white(self):
        if self.board.turn == chess.WHITE:
            return  self.pawn_extractor.isolated_pawns_color(chess.WHITE)
        return -self.pawn_extractor.isolated_pawns_color(chess.WHITE)

    def ft_isolated_pawns_black(self):
        if self.board.turn == chess.BLACK:
            return  self.pawn_extractor.isolated_pawns_color(chess.BLACK)
        return -self.pawn_extractor.isolated_pawns_color(chess.BLACK)


    def ft_king_ring_enemy_pressure(self) -> int:
        return self.king_extractor.king_ring_enemy_pressure()

    def ft_king_ring_enemy_pressure_white(self) -> int:
        if self.board.turn == chess.WHITE:
            return self.king_extractor.king_ring_enemy_pressure_color(chess.WHITE)
        return -self.king_extractor.king_ring_enemy_pressure_color(chess.WHITE)

    def ft_king_ring_enemy_pressure_black(self) -> int:
        if self.board.turn == chess.BLACK:
            return self.king_extractor.king_ring_enemy_pressure_color(chess.BLACK)
        return -self.king_extractor.king_ring_enemy_pressure_color(chess.BLACK)


    def ft_king_sqr_sum(self) -> int:
        if self.game_stage != END_GAME :
            return self.piece_sqr_sum(chess.KING)
        return self.piece_sqr_sum("king_end")

    def ft_king_sqr_sum_white(self):
        if self.board.turn == chess.WHITE:
            return  self.piece_sqr_sum_color(chess.WHITE, chess.KING, chess.KING)
        return -self.piece_sqr_sum_color(chess.WHITE, chess.KING, chess.KING)

    def ft_king_sqr_sum_black(self):
        if self.board.turn == chess.BLACK:
            return  self.piece_sqr_sum_color(chess.BLACK, chess.KING, chess.KING)
        return -self.piece_sqr_sum_color(chess.BLACK, chess.KING, chess.KING)


    def ft_knight_outposts_white(self):
        if self.board.turn == chess.WHITE:
            return self.knight_outposts(chess.WHITE)
        return -(self.knight_outposts(chess.WHITE))

    def ft_knight_outposts_black(self):
        if self.board.turn == chess.BLACK:
            return self.knight_outposts(chess.BLACK)
        return -(self.knight_outposts(chess.BLACK))


    def ft_knight_sqr_sum(self) -> int:
        return self.piece_sqr_sum(chess.KNIGHT)

    def ft_knight_sqr_sum_white(self):
        if self.board.turn == chess.WHITE:
            return  self.piece_sqr_sum_color(chess.WHITE, chess.KNIGHT, chess.KNIGHT)
        return -self.piece_sqr_sum_color(chess.WHITE, chess.KNIGHT, chess.KNIGHT)

    def ft_knight_sqr_sum_black(self):
        if self.board.turn == chess.BLACK:
            return  self.piece_sqr_sum_color(chess.BLACK, chess.KNIGHT, chess.KNIGHT)
        return -self.piece_sqr_sum_color(chess.BLACK, chess.KNIGHT, chess.KNIGHT)


    def ft_material_bishop_white(self):
        if self.board.turn == chess.WHITE:
            return self.material_balance_white[2]
        return -self.material_balance_white[2]

    def ft_material_bishop_black(self):
        if self.board.turn == chess.BLACK:
            return self.material_balance_black[2]
        return -self.material_balance_black[2]


    def ft_material_knight_white(self):
        if self.board.turn == chess.WHITE:
            return self.material_balance_white[1]
        return -self.material_balance_white[1]

    def ft_material_knight_black(self):
        if self.board.turn == chess.BLACK:
            return self.material_balance_black[1]
        return -self.material_balance_black[1]


    def ft_material_pawn_white(self):
        if self.board.turn == chess.WHITE:
            return self.material_balance_white[0]
        return -self.material_balance_white[0]

    def ft_material_pawn_black(self):
        if self.board.turn == chess.BLACK:
            return self.material_balance_black[0]
        return -self.material_balance_black[0]


    def ft_material_queen_white(self):
        if self.board.turn == chess.WHITE:
            return self.material_balance_white[4]
        return -self.material_balance_white[4]

    def ft_material_queen_black(self):
        if self.board.turn == chess.BLACK:
            return self.material_balance_black[4]
        return -self.material_balance_black[4]
        # if self.board.turn == chess.BLACK:
        #     return self.material_extractor.material_balance(chess.BLACK)[4]
        # return -self.material_extractor.material_balance(chess.BLACK)[4]

    def ft_material_rook_white(self):
        if self.board.turn == chess.WHITE:
            return self.material_balance_white[3]
        return -self.material_balance_white[3]

    def ft_material_rook_black(self):
        if self.board.turn == chess.BLACK:
            return self.material_balance_black[3]
        return -self.material_balance_black[3]

    def ft_mobility_balance(self):
        if self.board.turn == chess.WHITE:
            return sum(self.material_extractor.ft_mobility_balance())
        return -sum(self.material_extractor.ft_mobility_balance())

    def ft_mobility_white(self):
        if self.board.turn == chess.WHITE:
            return sum(self.material_extractor.mobility_color(chess.WHITE))
        return -sum(self.material_extractor.mobility_color(chess.WHITE))

    def ft_mobility_black(self):
        if self.board.turn == chess.BLACK:
            return sum(self.material_extractor.mobility_color(chess.BLACK))
        return -sum(self.material_extractor.mobility_color(chess.BLACK))

    def ft_mobility_safe_balance(self):
        if self.board.turn == chess.WHITE:
            return sum(self.material_extractor.ft_mobility_safe_balance())
        return -sum(self.material_extractor.ft_mobility_safe_balance())

    def ft_mobility_safe_white(self):
        if self.board.turn == chess.WHITE:
            return sum(self.material_extractor.mobility_color(chess.WHITE,
                                                              safety=True))
        return -sum(
            self.material_extractor.mobility_color(chess.WHITE, safety=True))

    def ft_mobility_safe_black(self):
        if self.board.turn == chess.BLACK:
            return sum(self.material_extractor.mobility_color(chess.BLACK,
                                                              safety=True))
        return -sum(
            self.material_extractor.mobility_color(chess.BLACK, safety=True))


    def ft_outposts_white(self) -> int:
        if self.board.turn == chess.WHITE:
            return self.bishop_outposts(chess.WHITE) + self.knight_outposts(chess.WHITE)
        return -(self.bishop_outposts(chess.WHITE) + self.knight_outposts(
            chess.WHITE))

    def ft_outposts_black(self) -> int:
        if self.board.turn == chess.BLACK:
            return self.bishop_outposts(chess.BLACK) + self.knight_outposts(chess.BLACK)
        return -(self.bishop_outposts(chess.BLACK) + self.knight_outposts(
            chess.BLACK))


    def ft_passed_pawns(self):
        return self.pawn_extractor.passed_pawns()

    def ft_passed_pawns_white(self):
        if self.board.turn == chess.WHITE:
            return  self.pawn_extractor.passed_pawns_color(chess.WHITE)
        return -self.pawn_extractor.passed_pawns_color(chess.WHITE)

    def ft_passed_pawns_black(self):
        if self.board.turn == chess.BLACK:
            return self.pawn_extractor.passed_pawns_color(chess.BLACK)
        return -self.pawn_extractor.passed_pawns_color(chess.BLACK)


    def ft_pawn_shield(self) -> int:
        return self.king_extractor.pawn_shield()

    def ft_pawn_shield_white(self) -> int:
        if self.board.turn == chess.WHITE:
            return self.king_extractor.pawn_shield_color(chess.WHITE)
        return -self.king_extractor.pawn_shield_color(chess.WHITE)

    def ft_pawn_shield_black(self) -> int:
        if self.board.turn == chess.BLACK:
            return self.king_extractor.pawn_shield_color(chess.BLACK)
        return -self.king_extractor.pawn_shield_color(chess.BLACK)


    def ft_pawn_sqr_sum(self) -> int:
        return self.piece_sqr_sum(chess.PAWN)

    def ft_pawn_sqr_sum_white(self):
        if self.board.turn == chess.WHITE:
            return  self.piece_sqr_sum_color(chess.WHITE, chess.PAWN, chess.PAWN)
        return -self.piece_sqr_sum_color(chess.WHITE, chess.PAWN, chess.PAWN)

    def ft_pawn_sqr_sum_black(self):
        if self.board.turn == chess.BLACK:
            return  self.piece_sqr_sum_color(chess.BLACK, chess.PAWN, chess.PAWN)
        return -self.piece_sqr_sum_color(chess.BLACK, chess.PAWN, chess.PAWN)


    def ft_pieces_occupying_center(self) -> np.array:
        pieces_occupy_center = np.zeros(6, dtype=float)
        for sq in CENTER:
            piece = self.board.piece_at(sq)
            if piece:
                piece_idx = piece.piece_type - 1  # 0=pawn, 1=knight, ..., 5=king
                # Always add if it's the player to move's piece, subtract if opponent's
                if piece.color == (
                        self.board.turn == chess.WHITE):  # player's piece
                    pieces_occupy_center[piece_idx] += 1
                else:  # opponent's piece
                    pieces_occupy_center[piece_idx] -= 1

        return np.sum(pieces_occupy_center * CENTER_OCCUPY_BONUS).item()


    def ft_queen_sqr_sum(self) -> int:
        return self.piece_sqr_sum(chess.QUEEN)

    def ft_queen_sqr_sum_white(self):
        if self.board.turn == chess.WHITE:
            return  self.piece_sqr_sum_color(chess.WHITE, chess.QUEEN, chess.QUEEN)
        return -self.piece_sqr_sum_color(chess.WHITE, chess.QUEEN, chess.QUEEN)

    def ft_queen_sqr_sum_black(self):
        if self.board.turn == chess.BLACK:
            return  self.piece_sqr_sum_color(chess.BLACK, chess.QUEEN, chess.QUEEN)
        return -self.piece_sqr_sum_color(chess.BLACK, chess.QUEEN, chess.QUEEN)


    def ft_rook_sqr_sum(self) -> int:
        return self.piece_sqr_sum(chess.ROOK)

    def ft_rook_sqr_sum_white(self):
        if self.board.turn == chess.WHITE:
            return  self.piece_sqr_sum_color(chess.WHITE, chess.ROOK, chess.ROOK)
        return -self.piece_sqr_sum_color(chess.WHITE, chess.ROOK, chess.ROOK)

    def ft_rook_sqr_sum_black(self):
        if self.board.turn == chess.BLACK:
            return  self.piece_sqr_sum_color(chess.BLACK, chess.ROOK, chess.ROOK)
        return -self.piece_sqr_sum_color(chess.BLACK, chess.ROOK, chess.ROOK)

    def sqr_sum_color(self, color):
        piece_types = [chess.PAWN, chess.BISHOP, chess.ROOK, chess.KNIGHT,
                       chess.KING, chess.QUEEN]
        ss = 0
        for p in piece_types:
            ss += self.piece_sqr_sum_color(color, p, p)
        return ss

    def ft_sqr_sum_black(self):
        if self.board.turn == chess.BLACK:
            return  self.sqr_sum_color(chess.BLACK)
        return -self.sqr_sum_color(chess.BLACK)

    def ft_sqr_sum_white(self):
        if self.board.turn == chess.WHITE:
            return  self.sqr_sum_color(chess.WHITE)
        return -self.sqr_sum_color(chess.WHITE)

    def ft_target(self) -> float:
        info = self.engine.analyse(self.board, chess.engine.Limit(nodes=1))
        score = info["score"].white()
        cp = score.score(mate_score=9999)/100.0

        return -cp if self.board.turn == chess.BLACK else cp


    def ft_threat_balance(self):
        if self.board.turn == chess.WHITE:
            return sum(self.material_extractor.ft_threat_balance())
        return -sum(self.material_extractor.ft_threat_balance())

    def ft_threat_white(self):
        if self.board.turn == chess.WHITE:
            return sum(self.material_extractor.threat(chess.WHITE))
        return -sum(self.material_extractor.threat(chess.WHITE))

    def ft_threat_black(self):
        if self.board.turn == chess.BLACK:
            return sum(self.material_extractor.threat(chess.BLACK))
        return -sum(self.material_extractor.threat(chess.BLACK))


    def outpost(self, piece_type: chess.PieceType, color: chess.Color)->int:
        outpost = 0
        for sqr in self.board.pieces(piece_type, color):
            if self.is_outpost(sqr, color, piece_type):
               outpost += 1
        return outpost

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
    def knight_outposts(self, color: chess.Color)->int:
        return self.outpost(chess.KNIGHT, color) *KNIGHT_OUTPOST_WEIGHT

    # count and subtract the number of bishop outposts white vs. black
    def bishop_outposts(self, color: chess.Color) ->int:
        return self.outpost(chess.BISHOP, color)*BISHOP_OUTPOST_WEIGHT

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

        ret = white_piece_sqr_sum - black_piece_sqr_sum
        if self.board.turn == chess.WHITE:
            return ret
        return -ret

    # call before any ft_material function to set the material value vectors for black and white
    def set_material_balances(self):
        self.material_balance_white = self.material_extractor.material_balance(chess.WHITE)
        self.material_balance_black = self.material_extractor.material_balance(chess.BLACK)

    def get_features(self):
        i = 0
        for name, method in inspect.getmembers(self, predicate=inspect.ismethod):
            if not name.startswith("ft_"): continue

            vec = method()

            if name.endswith("pieces_occupying_center") or name.endswith("center_attackers_white")or name.endswith("center_attackers_black"):
                self.features[i] = np.sum(vec* feature_weights[name[3:]]).item()
            elif isinstance(vec, int):
                if feature_weights[name[3:]] == 0:
                    self.features[i] = vec
                else:
                    self.features[i] = vec * feature_weights[name[3:]]
            else:
                self.features[i] = vec
            i += 1
        return self.features[:i]

    def get_features_subset_dict(self, feature_names):
        self.set_material_balances()
        result = {}
        for name in feature_names:
            if name not in self.feature_functions:
                result[name] = 0.0
            else:
                result[name] = self.feature_functions[name]()
        return result

def mirror_square(sq: int) -> int:
    return ((7 - (sq // 8)) * 8) + (sq % 8)


def write_features_file(stockfish_path, input_file='ann_group/positions.csv',output_file='ann_group/positions_with_features_NEW.csv'):
    fe = FeatureExtractorN(chess.Board(), 0, stockfish_path=stockfish_path)
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
    fe.engine.quit()


# features_names = [
#     # "attack_balance", #Y
#     "attack_black",
#     "attack_white",
#     # "bishop_outposts_black", # Z
#     # "bishop_outposts_white",
#     "bishop_pair_white", # early
#     "bishop_pair_black",
#     # "bishop_sqr_sum", #Y
#     # "bishop_sqr_sum_black",
#     # "bishop_sqr_sum_white",
#     "center_attackers_white", #Y
#     "center_attackers_black", #Y
#     "connected_pawns", # Y
#     # "connected_pawns_black",
#     # "connected_pawns_white",
#     "defense_balance", # Y
#     # "defense_black",
#     # "defense_white",
#     # "doubled_pawns", # zero
#     "doubled_pawns_black",
#     "doubled_pawns_white",
#     "half_open_king_files", # late
#     # "half_open_king_files_black",
#     # "half_open_king_files_white",
#     # "isolated_pawns", # mide - late
#     "isolated_pawns_black",
#     "isolated_pawns_white",
#     "king_ring_enemy_pressure", # Y
#     # "king_ring_enemy_pressure_black",
#     # "king_ring_enemy_pressure_white",
#     # "king_sqr_sum",
#     # "king_sqr_sum_black", # Y
#     # "king_sqr_sum_white", # Y
#     # "knight_outposts_black", # zero
#     # "knight_outposts_white", # zero
#     # "knight_sqr_sum", # early - mid
#     # "knight_sqr_sum_black",
#     # "knight_sqr_sum_white",
#     "material_bishop_white", # early - mid
#     "material_bishop_black", # early - mid
#     "material_knight_white", # early - mid
#     "material_knight_black", # early - mid
#     "material_pawn_white", # Y
#     "material_pawn_black", # Y
#     "material_queen_white", # early - mid
#     "material_queen_black", # early - mid
#     "material_rook_white", # early - mid
#     "material_rook_black", # early - mid
#     # "mobility_balance", # Y
#     "mobility_black",
#     "mobility_white",
#     # "mobility_safe_balance", # Y
#     "mobility_safe_black",
#     "mobility_safe_white",
#     "outposts_white", # zero
#     "outposts_black", # zero
#     "passed_pawns", # mid - late
#     # "passed_pawns_black",
#     # "passed_pawns_white",
#     "pawn_shield", # Y
#     # "pawn_shield_black",
#     # "pawn_shield_white",
#     # "pawn_sqr_sum", # Y
#     # "pawn_sqr_sum_black",
#     # "pawn_sqr_sum_white",
#     "pieces_occupying_center", # Y
#     # "queen_sqr_sum", # zero
#     # "queen_sqr_sum_black",
#     # "queen_sqr_sum_white",
#     # "rook_sqr_sum", # zero
#     # "rook_sqr_sum_black",
#     # "rook_sqr_sum_white",
#     "sqr_sum_black",
#     "sqr_sum_white",
#     # "threat_balance",
#     "threat_black", # Y
#     "threat_white" # Y
# ]

# fe = FeatureExtractorN(chess.Board("rnbqkbnr/ppp2ppp/4p3/3p4/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 3"), EARLY_GAME)
# print(fe.get_features_subset_dict(features_names))
# fe.engine.quit()