import chess, time, csv
from mcts import mcts_search

MCTS_ITERS = 600  

def play_selfplay():
    board = chess.Board()
    moves = 0

    while not board.is_game_over() and moves < 200:
        mv = mcts_search(board, iters=MCTS_ITERS)
        board.push(mv)
        moves += 1
        print(board, "\n")

    return board.result(), moves


def run_matches(n=5, csv_path="results.csv"):
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["game", "result", "moves", "time(sec)"])

        for i in range(n):
            print(f"\n========== Game {i+1} ==========")
            t0 = time.time()

            result, m = play_selfplay()
            dt = time.time() - t0

            print(f"Game {i+1} finished — result={result}, moves={m}, time={dt:.1f}s")

            w.writerow([i + 1, result, m, f"{dt:.1f}"])


if __name__ == "__main__":
    run_matches()
