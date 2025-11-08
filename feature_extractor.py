import chess
import chess.engine
import numpy as np

PAWN_SQR_TABLE = [
    0, 0, 0, 0, 0, 0, 0, 0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
    5, 5, 10, 25, 25, 10, 5, 5,
    0, 0, 0, 20, 20, 0, 0, 0,
    5, -5, -10, 0, 0, -10, -5, 5,
    5, 10, 10, -20, -20, 10, 10, 5,
    0, 0, 0, 0, 0, 0, 0, 0
]

KNIGHT_SQR_TABLE = [
    -50, -40, -30, -30, -30, -30, -40, -50,
    -40, -20, 0, 0, 0, 0, -20, -40,
    -30, 0, 10, 15, 15, 10, 0, -30,
    -30, 5, 15, 20, 20, 15, 5, -30,
    -30, 0, 15, 20, 20, 15, 0, -30,
    -30, 5, 10, 15, 15, 10, 5, -30,
    -40, -20, 0, 5, 5, 0, -20, -40,
    -50, -40, -30, -30, -30, -30, -40, -50,
]

BISHOP_SQR_TABLE = [
    -20, -10, -10, -10, -10, -10, -10, -20,
    -10, 0, 0, 0, 0, 0, 0, -10,
    -10, 0, 5, 10, 10, 5, 0, -10,
    -10, 5, 5, 10, 10, 5, 5, -10,
    -10, 0, 10, 10, 10, 10, 0, -10,
    -10, 10, 10, 10, 10, 10, 10, -10,
    -10, 5, 0, 0, 0, 0, 5, -10,
    -20, -10, -10, -10, -10, -10, -10, -20,
]

ROOK_SQR_TABLE = [
    0, 0, 0, 0, 0, 0, 0, 0,
    5, 10, 10, 10, 10, 10, 10, 5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    0, 0, 0, 5, 5, 0, 0, 0
]

QUEEN_SQR_TABLE = [
    -20, -10, -10, -5, -5, -10, -10, -20,
    -10, 0, 0, 0, 0, 0, 0, -10,
    -10, 0, 5, 5, 5, 5, 0, -10,
    -5, 0, 5, 5, 5, 5, 0, -5,
    0, 0, 5, 5, 5, 5, 0, -5,
    -10, 5, 5, 5, 5, 5, 0, -10,
    -10, 0, 5, 0, 0, 0, 0, -10,
    -20, -10, -10, -5, -5, -10, -10, -20
]
KING_MID_SQR_TABLE = [
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -20, -30, -30, -40, -40, -30, -30, -20,
    -10, -20, -20, -20, -20, -20, -20, -10,
    20, 20, 0, 0, 0, 0, 20, 20,
    20, 30, 10, 0, 0, 10, 30, 20
]
KING_LATE_SQR_TABLE = [
    -50, -40, -30, -20, -20, -30, -40, -50,
    -30, -20, -10, 0, 0, -10, -20, -30,
    -30, -10, 20, 30, 30, 20, -10, -30,
    -30, -10, 30, 40, 40, 30, -10, -30,
    -30, -10, 30, 40, 40, 30, -10, -30,
    -30, -10, 20, 30, 30, 20, -10, -30,
    -30, -30, 0, 0, 0, 0, -30, -30,
    -50, -30, -30, -30, -30, -30, -30, -50
]

CENTER = {chess.D4, chess.E4, chess.D5, chess.E5}


class FeatureExtractor:
    def __init__(self, board: chess.Board):
        self.board = board

    # count and subtract number of white vs black pieces occupying the center
    def pieces_occupying_enter(self):
        white_occ_count = 0
        black_occ_count = 0
        for sq in CENTER:
            if self.board.piece_at(sq).color == chess.WHITE:
                white_occ_count += 1
            elif self.board.piece_at(sq).color == chess.BLACK:
                black_occ_count += 1
        if self.board.turn == chess.WHITE:
            return white_occ_count - black_occ_count
        return black_occ_count - white_occ_count

    #  count and subtract number of white vs black pieces attacking the center
    def center_attackers(self):
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
    def bishop_pair(self):
        white_has_bp = 1 if len(self.board.pieces(chess.BISHOP, chess.WHITE)) == 2 else 0
        black_has_bp = 1 if len(self.board.pieces(chess.BISHOP, chess.BLACK)) == 2 else 0
        if self.board.turn == chess.WHITE:
            return white_has_bp - black_has_bp
        return black_has_bp - white_has_bp

    def outpost(self, piece_type: chess.PieceType):
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
    def is_outpost(self, sqr: chess.Square, color: chess.Color, piece_type: chess.PieceType):
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
    def knight_outposts(self):
        return self.outpost(chess.KNIGHT)

    # count and subtract the number of bishop outposts white vs. black
    def bishop_outposts(self):
        return self.outpost(chess.BISHOP)


    def passedPawns(self):
        # TODO
        return


    def pawnSqrTableSum(self):
        # TODO
        return


    def knightSqrTableSum(self):
        # TODO
        return


    def bishopSqrTableSum(self):
        # TODO
        return


    def rookSqrTableSum(self):
        # TODO
        return


    def queenSqrTableSum(self):
        # TODO
        return


    def kingMiddleSqrTableSum(self):
        # TODO
        return


    def kingLateSqrTableSum(self):
        # TODO
        return

        # CenterControl
        # cc[] = {12,12,18,24}
        # bishop pair
        # if (white_bishops >= 2) score += 45;
        # if (black_bishops >= 2) score -= 45;
        # Outposts (Knight/Bishop) (+30)
        # if (knight on e5/d5 && supported by own pawn) score += 35;
        # if (bishop on d6/e6 && supported) score += 28;
        # Passed Pawns (Beyond Structure) (+40 Elo)
        # if (is_passed(pawn)) score += 80 / distance_to_promotion;

