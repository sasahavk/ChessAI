import chess
import time
import csv

from mcts_nomem_stats import StatsMCTSBot
from minimax_group_evaluate import evaluate

MCTS_ITERS = 150
MAX_DEPTH = 60
MAX_MOVES = 200
N_GAMES = 5

def simulate_game(bot_white, bot_black):
    board = chess.Board()
    moves = 0
    move_times = []
    sim_depths = []
    root_sims = []
    non_win_evals = []
    while not board.is_game_over() and moves < MAX_MOVES:
        bot = bot_white if board.turn == chess.WHITE else bot_black
        t0 = time.time()
        move, stats = bot.play(board)
        dt = time.time() - t0
        move_times.append(dt)
        sim_depths.append(stats["avg_sim_depth"])
        root_sims.append(stats["root_simulations"])
        non_win_evals.append(stats["avg_non_win_eval"])
        board.push(move)
        moves += 1
    result = board.result(claim_draw=True)
    if result not in ["1-0", "0-1", "1/2-1/2"]:
        result = "*"
    avg_time = sum(move_times) / len(move_times) if move_times else 0.0
    avg_sim_depth = sum(sim_depths) / len(sim_depths) if sim_depths else 0.0
    avg_root_sims = sum(root_sims) / len(root_sims) if root_sims else 0.0
    avg_non_win_eval = sum(non_win_evals) / len(non_win_evals) if non_win_evals else 0.0
    return {
        "result": result,
        "moves": moves,
        "avg_time": avg_time,
        "avg_sim_depth": avg_sim_depth,
        "avg_root_sims": avg_root_sims,
        "avg_non_win_eval": avg_non_win_eval
    }

def run_self_play(csv_path="results_self_nomem_stats.csv"):
    bot_white = StatsMCTSBot(
        numRootSimulations=MCTS_ITERS,
        maxSimDepth=MAX_DEPTH,
        evalFunc=evaluate
    )
    bot_black = StatsMCTSBot(
        numRootSimulations=MCTS_ITERS,
        maxSimDepth=MAX_DEPTH,
        evalFunc=evaluate
    )
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "game",
            "result",
            "moves",
            "avg_time",
            "avg_sim_depth",
            "avg_root_sims",
            "avg_non_win_eval",
            "opponent",
            "mem"
        ])
        for i in range(N_GAMES):
            stats = simulate_game(bot_white, bot_black)
            print("game", i + 1, "done:", stats)
            w.writerow([
                i + 1,
                stats["result"],
                stats["moves"],
                stats["avg_time"],
                stats["avg_sim_depth"],
                stats["avg_root_sims"],
                stats["avg_non_win_eval"],
                "mcts_self",
                "no-mem"
            ])

if __name__ == "__main__":
    run_self_play()
