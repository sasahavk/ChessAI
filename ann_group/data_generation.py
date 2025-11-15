import random, csv, os, chess, chess.engine
from feature_extractor_material import MaterialFeatureExtractor

OUT_CSV = ""
TARGET_ROWS = 33

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
        "Skill Level": random.randint(0, 20),
    })
    total_rows, games = 0, 0

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        try:
            while total_rows < TARGET_ROWS:
                board = chess.Board()
                plies = 0
                game_rows = []
                while not board.is_game_over(claim_draw=True) and plies < 120:
                    state = board.fen()
                    turn = int(board.turn)
                    move_num = min(board.fullmove_number, 60)
                    features = MaterialFeatureExtractor(board, int(board.fullmove_number / 20)).extract_features()
                    game_rows.append([state, turn, move_num, None] + features)
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

if __name__ == "__main__": main()
