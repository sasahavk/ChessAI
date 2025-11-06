import csv, os, chess, chess.engine

OUT_CSV = ""
TARGET_ROWS = 33000

PIECE_VALUE = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,
}

def material_balance(board: chess.Board):
    score = 0
    for piece_type, v in PIECE_VALUE.items():
        score += v * (len(board.pieces(piece_type, chess.WHITE)) - len(board.pieces(piece_type, chess.BLACK)))
    if board.turn == chess.BLACK: score *= -1
    return score

def mobility(board: chess.Board):
    score = board.legal_moves.count()
    board.push(chess.Move.null())
    score -= board.legal_moves.count()
    board.pop()
    return score

def encode_result_for_side(final_result: str, side_to_move: int) -> int:
    if final_result in ("1/2-1/2", "*"): return 1
    if final_result == "1-0":
        if side_to_move == 1: return 2
        return 0
    if final_result == "0-1":
        if side_to_move == 0: return 2
        return 0
    return 1

def main():
    os.makedirs(os.path.dirname(OUT_CSV) or ".", exist_ok=True)
    engine = chess.engine.SimpleEngine.popen_uci(r"C:\cs\stockfish\stockfish-windows-x86-64-avx2.exe")
    engine.configure({
        "Skill Level": 2,
    })
    total_rows, games = 0, 0

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["board", "turn", "move_num", "result", "material_balance", "mobility"])
        try:
            while total_rows < TARGET_ROWS:
                board = chess.Board()
                plies = 0
                game_rows = []
                while not board.is_game_over(claim_draw=True) and plies < 120:
                    state = board.fen()
                    turn = int(board.turn)
                    move_num = min(board.fullmove_number, 60)
                    material_balance_v = material_balance(board)
                    mobility_v = mobility(board)
                    game_rows.append([state, turn, move_num, None, material_balance_v, mobility_v])
                    result = engine.play(board, chess.engine.Limit(time=0.01))
                    if result.move is None:
                        break
                    board.push(result.move)
                    plies += 1
                if board.fullmove_number > 60 or not board.is_game_over(claim_draw=True): final = "1/2-1/2"
                else: final = board.result(claim_draw=True)
                for row in game_rows:
                    row[3] = encode_result_for_side(final, row[1])
                    writer.writerow(row)
                    total_rows += 1
                games += 1
        finally:
            engine.quit()

main()
