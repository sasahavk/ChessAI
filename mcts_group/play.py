import chess, time, csv
import chess.engine
from mcts_bot import MonteCarloSearchTreeBot
from minimax_group_evaluate import evaluate
# from evaluate import evaluate

MCTS_ITERS:int = 800
MAX_DEPTH:int = 150
MAX_MOVES:int = 300

STOCKFISH_LIMIT = chess.engine.Limit(time=0.1)  # or depth=12, nodes=...
STOCKFISH_ELO_DEFAULT = 1500
STOCKFISH_ELO_MIN = 1320
STOCKFISH_PATH = r"/home/thielith/Desktop/school/ECS_170/project/stockfish_bmi2/stockfish-ubuntu-x86-64-bmi2"

stockfish = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
stockfish.configure({"UCI_LimitStrength": True})

def simulateGame(bot1, bot2):
    board = chess.Board()
    # board.set_board_fen("k7/2K5/8/1Q6/8/8/8/8")  # one move away from a win on white's side
    moves = 0

    bot1Stats:dict = None
    bot2Stats:dict = None
    while not board.is_game_over() and moves < MAX_MOVES:
        if moves % 2 == 0:
            bot1Stats = performMove(board, bot1, moves)
        else:
            bot2Stats = performMove(board, bot2, moves)
        moves += 1

    return board.result(), moves, bot1Stats, bot2Stats

def performMove(board, bot, moveID=-1):
    timeStart = time.time()

    result = None
    stats:dict = None
    if bot == stockfish:
        result = bot.play(board, STOCKFISH_LIMIT)
    elif type(bot) == MonteCarloSearchTreeBot:
        result, stats = bot.play(board)
        
    if type(result) == chess.engine.PlayResult:
        board.push(result.move)
    else:
        board.push(result)
    print(board, "\n")

    timeEnd = time.time()
    timeTaken = timeEnd - timeStart

    print(f"Time taken for move {moveID}:  {timeTaken} seconds")
    print("===================")

    return stats

def run_matches(bot1, bot2, numMatches=5, fileName="results"):
    with open(f"./results/{fileName}.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["game", "winner", "moves", "time(sec)",
            "avgSimDepth", "numRootSim", "avgBoardEvalScore", "numFoundWinningMoveSets"
        ])

        for i in range(numMatches):
            print(f"\n========== Game {i+1} ==========")
            startTime = time.time()

            result, numMoves, bot1Stats, bot2Stats = simulateGame(bot1, bot2)
            totalTime = time.time() - startTime

            if result == "1-0":  result = "white"
            elif result == "0-1":  result = "black"
            elif result == "1/2-1/2":  result = "tie"
            else:  result == None

            if bot1Stats != None:
                print("bot1:")
                print(bot1Stats)
            if bot2Stats != None:
                print("bot2:")
                print(bot2Stats)

            stats = bot1Stats if bot1Stats != None else bot2Stats
            if bot1Stats and bot2Stats:
                stats = {
                    "avgSimDepth": (bot1Stats["avgSimDepth"] + bot2Stats["avgSimDepth"]) / 2.0,
                    "numRootSims": (bot1Stats["numRootSims"] + bot2Stats["numRootSims"]) / 2.0,
                    "boardEvalScore": (bot1Stats["boardEvalScore"] + bot2Stats["boardEvalScore"]) / 2.0,
                    "foundWinningMoveSets": (bot1Stats["foundWinningMoveSets"] + bot2Stats["foundWinningMoveSets"]) / 2.0
                }

            print(f"Game {i+1} finished - winner={result}, moves={numMoves}, time={totalTime:.1f}s, ")
            print(f"\tavg sim depth: {stats["avgSimDepth"]}, num root sims: {stats["numRootSims"]}, ")
            print(f"\tavg non-end eval score: {stats["boardEvalScore"]}, num found winning movesets: {stats["foundWinningMoveSets"]}")

            writer.writerow([i+1, result, numMoves, f"{totalTime:.1f}",
                stats["avgSimDepth"], stats["numRootSims"], stats["boardEvalScore"], stats["foundWinningMoveSets"]
            ])
        
        stockfish.quit()

def runSeveralConfigdMatches(numPerBatch:int, botOpponent=None, botName:str=""):
    mctsBotDefault = MonteCarloSearchTreeBot(
        numRootSimulations=MCTS_ITERS,
        maxSimDepth=MAX_DEPTH,
        evalFunc=evaluate,
        rememberPastBoardScores=False,
        conditionForSimulatingTriedMoves=lambda n: False,
        goForWin=False
    )
    mctsBotRemember = MonteCarloSearchTreeBot(
        numRootSimulations=MCTS_ITERS,
        maxSimDepth=MAX_DEPTH,
        evalFunc=evaluate,
        rememberPastBoardScores=True,
        conditionForSimulatingTriedMoves=lambda n: False,
        goForWin=False
    )
    mctsBotSimTriedMoves = MonteCarloSearchTreeBot(
        numRootSimulations=MCTS_ITERS,
        maxSimDepth=MAX_DEPTH,
        evalFunc=evaluate,
        rememberPastBoardScores=False,
        conditionForSimulatingTriedMoves=None,
        goForWin=False
    )
    mctsBotJumpTheWinGun = MonteCarloSearchTreeBot(
        numRootSimulations=MCTS_ITERS,
        maxSimDepth=MAX_DEPTH,
        evalFunc=evaluate,
        rememberPastBoardScores=False,
        conditionForSimulatingTriedMoves=lambda n: False,
        goForWin=True
    )

    mctsBotRemember_SimTriedMoves = MonteCarloSearchTreeBot(
        numRootSimulations=MCTS_ITERS,
        maxSimDepth=MAX_DEPTH,
        evalFunc=evaluate,
        rememberPastBoardScores=True,
        conditionForSimulatingTriedMoves=None,
        goForWin=False
    )
    mctsBotRemember_JumpTheWinGun = MonteCarloSearchTreeBot(
        numRootSimulations=MCTS_ITERS,
        maxSimDepth=MAX_DEPTH,
        evalFunc=evaluate,
        rememberPastBoardScores=True,
        conditionForSimulatingTriedMoves=lambda n: False,
        goForWin=True
    )

    mctsBotSimTried_JumpWin = MonteCarloSearchTreeBot(
        numRootSimulations=MCTS_ITERS,
        maxSimDepth=MAX_DEPTH,
        evalFunc=evaluate,
        rememberPastBoardScores=False,
        conditionForSimulatingTriedMoves=None,
        goForWin=True
    )

    mctsBotAllThree = MonteCarloSearchTreeBot(
        numRootSimulations=MCTS_ITERS,
        maxSimDepth=MAX_DEPTH,
        evalFunc=evaluate,
        rememberPastBoardScores=True,
        conditionForSimulatingTriedMoves=None,
        goForWin=True
    )

    # against self
    if botOpponent == None:
        run_matches(bot1=mctsBotDefault, bot2=mctsBotDefault, numMatches=numPerBatch, fileName="results_default_self")
        run_matches(bot1=mctsBotRemember, bot2=mctsBotRemember, numMatches=numPerBatch, fileName="results_remember_self")
        run_matches(bot1=mctsBotSimTriedMoves, bot2=mctsBotSimTriedMoves, numMatches=numPerBatch, fileName="results_lookAtTriedMoves_self")
        run_matches(bot1=mctsBotJumpTheWinGun, bot2=mctsBotJumpTheWinGun, numMatches=numPerBatch, fileName="results_jumpTheWinGun_self")

        run_matches(bot1=mctsBotRemember_SimTriedMoves, bot2=mctsBotRemember_SimTriedMoves, numMatches=numPerBatch, fileName="results_remember-simTried_self")
        run_matches(bot1=mctsBotRemember_JumpTheWinGun, bot2=mctsBotRemember_JumpTheWinGun, numMatches=numPerBatch, fileName="results_remember-win_self")
        run_matches(bot1=mctsBotSimTried_JumpWin, bot2=mctsBotSimTried_JumpWin, numMatches=numPerBatch, fileName="results_simTried-win_self")
        run_matches(bot1=mctsBotAllThree, bot2=mctsBotAllThree, numMatches=numPerBatch, fileName="results_allThree_self")
        return

    # against bot, is white
    run_matches(bot1=mctsBotDefault, bot2=botOpponent, numMatches=numPerBatch, fileName=f"results_default_asWhite_{botName}")
    run_matches(bot1=mctsBotRemember, bot2=botOpponent, numMatches=numPerBatch, fileName=f"results_remember_asWhite_{botName}")
    run_matches(bot1=mctsBotSimTriedMoves, bot2=botOpponent, numMatches=numPerBatch, fileName=f"results_lookAtTriedMoves_asWhite_{botName}")
    run_matches(bot1=mctsBotJumpTheWinGun, bot2=botOpponent, numMatches=numPerBatch, fileName=f"results_jumpTheWinGun_asWhite_{botName}")

    run_matches(bot1=mctsBotRemember_SimTriedMoves, bot2=botOpponent, numMatches=numPerBatch, fileName=f"results_remember-simTried_asWhite_{botName}")
    run_matches(bot1=mctsBotRemember_JumpTheWinGun, bot2=botOpponent, numMatches=numPerBatch, fileName=f"results_remember-win_asWhite_{botName}")
    run_matches(bot1=mctsBotSimTried_JumpWin, bot2=botOpponent, numMatches=numPerBatch, fileName=f"results_simTried-win_asWhite_{botName}")
    run_matches(bot1=mctsBotAllThree, bot2=botOpponent, numMatches=numPerBatch, fileName=f"results_allThree_asWhite_{botName}")

    # against bot, is black
    run_matches(bot2=mctsBotDefault, bot1=botOpponent, numMatches=numPerBatch, fileName=f"results_default_asBlack_{botName}")
    run_matches(bot2=mctsBotRemember, bot1=botOpponent, numMatches=numPerBatch, fileName=f"results_remember_asBlack_{botName}")
    run_matches(bot2=mctsBotSimTriedMoves, bot1=botOpponent, numMatches=numPerBatch, fileName=f"results_lookAtTriedMoves_asBlack_{botName}")
    run_matches(bot2=mctsBotJumpTheWinGun, bot1=botOpponent, numMatches=numPerBatch, fileName=f"results_jumpTheWinGun_asBlack_{botName}")

    run_matches(bot2=mctsBotRemember_SimTriedMoves, bot1=botOpponent, numMatches=numPerBatch, fileName=f"results_remember-simTried_asBlack_{botName}")
    run_matches(bot2=mctsBotRemember_JumpTheWinGun, bot1=botOpponent, numMatches=numPerBatch, fileName=f"results_remember-win_asBlack_{botName}")
    run_matches(bot2=mctsBotSimTried_JumpWin, bot1=botOpponent, numMatches=numPerBatch, fileName=f"results_simTried-win_asBlack_{botName}")
    run_matches(bot2=mctsBotAllThree, bot1=botOpponent, numMatches=numPerBatch, fileName=f"results_allThree_asBlack_{botName}")
    


if __name__ == "__main__":
    # run_matches(numMatches=10)
    runSeveralConfigdMatches(10)
    stockfish.configure({"UCI_Elo": STOCKFISH_ELO_DEFAULT})
    runSeveralConfigdMatches(10, botOpponent=stockfish, botName=f"stockfish-{STOCKFISH_ELO_DEFAULT}")
    stockfish.configure({"UCI_Elo": STOCKFISH_ELO_MIN})
    runSeveralConfigdMatches(10, botOpponent=stockfish, botName=f"stockfish-{STOCKFISH_ELO_MIN}")



