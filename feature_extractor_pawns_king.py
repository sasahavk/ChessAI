import pandas as pd
import chess

CSV_PATH = "./games_new.csv"
N_ROWS   = 20

EARLY_GAME = 0
MID_GAME   = 1
END_GAME   = 2

# weights
PASSED_PAWN_WEIGHT = 10
DOUBLED_PAWN_WEIGHT = -20
ISOLATED_PAWN_WEIGHT = -20
CONNECTED_PAWN_WEIGHT = 8

KING_SHIELD_WEIGHT = 10    
HALF_OPEN_KING_FILES_PEN = -12
KING_RING_PRESSURE_WEIGHT = -3     

# Attacker weights for ring pressure
ATTACKER_WEIGHT = {
    chess.PAWN: 1,
    chess.KNIGHT: 2,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 5,
    chess.KING: 0,
}

def determine_stage(move_num: int) -> int:
    if move_num <= 15:
        return EARLY_GAME
    if move_num <= 35:
        return MID_GAME
    return END_GAME


class PawnFeatureExtractor:
    def __init__(self, board: chess.Board, game_stage: int):
        self.board = board
        self.feature_count = 18
        self.features = [0 for _ in range(self.feature_count)]
        self.game_stage = game_stage

    def set_board(self, board):
        self.board = board

    def passed_pawns(self):
        white_pawns = self.board.pieces(chess.PAWN, chess.WHITE)
        white_passed_pawns = 0
        for p in chess.SquareSet(white_pawns):
            if self.is_passed_pawn(p, chess.WHITE):
                white_passed_pawns += 1

        black_pawns = self.board.pieces(chess.PAWN, chess.BLACK)
        black_passed_pawns = 0
        for p in chess.SquareSet(black_pawns):
            if self.is_passed_pawn(p, chess.BLACK):
                black_passed_pawns += 1

        if self.board.turn == chess.WHITE:
            return white_passed_pawns - black_passed_pawns
        return black_passed_pawns - white_passed_pawns

    def is_passed_pawn(self, sqr: chess.Square, color: chess.Color):
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

    def compute(self) -> dict:
        passed = self.passed_pawns() * PASSED_PAWN_WEIGHT
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


class KingFeatureExtractor:
    def __init__(self, board: chess.Board, game_stage: int):
        self.board = board
        self.game_stage = game_stage

    def set_board(self, board):
        self.board = board

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

    def compute(self) -> dict:
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


def run_csv(csv_path, n_rows=20):
    df = pd.read_csv(csv_path, nrows=n_rows)
    rows = []

    for _, row in df.iterrows():
        fen = str(row.get("board", "")).strip()
        move_num = int(row.get("move_num", 1))

        # determine stage
        game_stage = determine_stage(move_num)

        board = None
        if fen:
            try:
                board = chess.Board(fen)
            except Exception:
                pass
        if board is None:
            # skip bad row
            continue

        pawn_fx = PawnFeatureExtractor(board, game_stage).compute()
        king_fx = KingFeatureExtractor(board, game_stage).compute()

        rows.append({
            "move_num": move_num,
            "turn": row.get("turn"),
            "result": row.get("result"),
            "stage": game_stage,
            **pawn_fx,
            **king_fx,
        })

    out = pd.DataFrame(rows)
    print(out)
    out.to_csv("features_sample.csv", index=False)
    print("\nSaved to features_sample.csv")


if __name__ == "__main__":
    run_csv(CSV_PATH, N_ROWS)

