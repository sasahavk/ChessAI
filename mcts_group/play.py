import chess, time, csv
import chess.engine
from mcts_bot import MonteCarloSearchTreeBot
# from minimax_group_evaluate import evaluate
from evaluate import evaluate

MCTS_ITERS:int = 1000
MAX_DEPTH:int = 150
MAX_MOVES:int = 200

STOCKFISH_LIMIT = chess.engine.Limit(time=0.1)  # or depth=12, nodes=...
STOCKFISH_ELO = 1500
STOCKFISH_PATH = r""

stockfish = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
stockfish.configure({"UCI_LimitStrength": True, "UCI_Elo": STOCKFISH_ELO})

def simulateGame(bot1, bot2):
    board = chess.Board()
    moves = 0

    while not board.is_game_over() and moves < MAX_MOVES:
        if moves % 2 == 0:
            performMove(board, bot1, moves)
        else:
            performMove(board, bot2, moves)
        moves += 1

    return board.result(), moves

def performMove(board, bot, moveID=-1):
    timeStart = time.time()

    result = bot.play(board) if not (bot == stockfish) else bot.play(board, STOCKFISH_LIMIT)
    if type(result) == chess.engine.PlayResult:
        board.push(result.move)
    else:
        board.push(result)
    print(board, "\n")

    timeEnd = time.time()
    timeTaken = timeEnd - timeStart
    print(f"Time taken for move {moveID}:  {timeTaken} seconds")
    print("===================")

def run_matches(n=5, csv_path="./results/results.csv"):
    mctsBot = MonteCarloSearchTreeBot(
        numRootSimulations=MCTS_ITERS,
        maxSimDepth=MAX_DEPTH,
        evalFunc=evaluate
    )

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["game", "result", "moves", "time(sec)"])

        for i in range(n):
            print(f"\n========== Game {i+1} ==========")
            t0 = time.time()

            result, m = simulateGame(mctsBot, stockfish)
            dt = time.time() - t0

            print(f"Game {i+1} finished — result={result}, moves={m}, time={dt:.1f}s")

            w.writerow([i + 1, result, m, f"{dt:.1f}"])
        
        stockfish.quit()


if __name__ == "__main__":
    run_matches(n=5)



