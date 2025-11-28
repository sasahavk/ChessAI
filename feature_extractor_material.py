import inspect, chess, chess.engine

PIECE_VALUE = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,
}

EARLY_GAME = 0
MID_GAME = 1
END_GAME = 2


class MaterialFeatureExtractor:
    def __init__(self, board: chess.Board, game_stage: int):
        self.board = board
        self.feature_count = 65
        self.features = [0] * self.feature_count
        self.game_stage = game_stage

    def set_board(self, board):
        self.board = board

    def extract_features(self):
        i = 0
        for name, method in sorted(inspect.getmembers(self, predicate=inspect.ismethod)):
            if not name.startswith("ft_"): continue
            vec = method()
            for value in vec:
                self.features[i] = value
                print(value)
                i += 1
        return self.features

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