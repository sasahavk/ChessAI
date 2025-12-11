import chess, time, csv
import chess.engine
from mcts_bot import MonteCarloSearchTreeBot
from minimax_group_evaluate import evaluate
# from evaluate import evaluate

MCTS_ITERS:int = 800
MAX_DEPTH:int = 150
MAX_MOVES:int = 300

STOCKFISH_LIMIT = chess.engine.Limit(time=0.1)  # or depth=12, nodes=...
STOCKFISH_ELO_DEFAULT = 2000
STOCKFISH_ELO_MIN = 1320
STOCKFISH_PATH = r""

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
    # print(board, "\n")

    timeEnd = time.time()
    timeTaken = timeEnd - timeStart

    if stats != None:
        stats["timePerMove"] = timeTaken

    # print(f"Time taken for move {moveID}:  {timeTaken} seconds")
    # print("===================")

    return stats

def run_matches(bot1, bot2, numMatches=5, fileName="results"):
    print(f"generating results in ./results/{fileName}.csv")

    with open(f"./results/{fileName}.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["game", "winner", "moves", "time(sec)", "timePerMove",
            "avgSimDepth", "numRootSim", "avgBoardEvalScore", "numFoundWinningMoveSets"
        ])

        numMovesFromAll = []
        gameTimes = []
        moveTimes = []
        simDepths = []
        rootSims = []
        boardEvals = []
        numWinMoveSets = []

        for i in range(numMatches):
            print(f"\n========== Game {i+1} ==========")
            startTime = time.time()

            result, numMoves, bot1Stats, bot2Stats = simulateGame(bot1, bot2)
            totalTime = time.time() - startTime

            if result == "1-0":  result = "white"
            elif result == "0-1":  result = "black"
            elif result == "1/2-1/2":  result = "tie"

            # if bot1Stats != None:
            #     print("bot1:")
            #     print(bot1Stats)
            # if bot2Stats != None:
            #     print("bot2:")
            #     print(bot2Stats)

            stats = bot1Stats if bot1Stats != None else bot2Stats
            if bot1Stats and bot2Stats:
                if bot1Stats["boardEvalScore"] != None and bot2Stats["boardEvalScore"] != None:
                    stats["boardEvaleScore"] = (bot1Stats["boardEvalScore"] + bot2Stats["boardEvalScore"]) / 2.0
                
                stats["avgSimDepth"] = (bot1Stats["avgSimDepth"] + bot2Stats["avgSimDepth"]) / 2.0
                stats["numRootSims"] = (bot1Stats["numRootSims"] + bot2Stats["numRootSims"]) / 2.0
                stats["foundWinningMoveSets"] = (bot1Stats["foundWinningMoveSets"] + bot2Stats["foundWinningMoveSets"]) / 2.0
                stats["timePerMove"] = (bot1Stats["timePerMove"] + bot2Stats["timePerMove"]) / 2.0

            print(f"Game {i+1} finished - winner={result}, moves={numMoves}, time={totalTime:.1f}s, avg move time={stats["timePerMove"]:.1f}")
            print(f"\tavg sim depth: {stats["avgSimDepth"]}, num root sims: {stats["numRootSims"]}, ")
            print(f"\tavg non-end eval score: {stats["boardEvalScore"]}, num found winning movesets: {stats["foundWinningMoveSets"]}")

            writer.writerow([i+1, result, numMoves, f"{totalTime:.1f}", stats["timePerMove"],
                stats["avgSimDepth"], stats["numRootSims"], stats["boardEvalScore"], stats["foundWinningMoveSets"]
            ])

            numMovesFromAll.append(numMoves)
            gameTimes.append(totalTime)
            moveTimes.append(stats["timePerMove"])
            simDepths.append(stats["avgSimDepth"])
            rootSims.append(stats["numRootSims"])
            boardEvals.append(stats["boardEvalScore"])
            numWinMoveSets.append(stats["foundWinningMoveSets"])
        
        # average all entries
        avgNumMoves = sum(numMovesFromAll) / len(numMovesFromAll)
        avgGameTimes = sum(gameTimes) / len(gameTimes)
        avgMoveTimes = sum(moveTimes) / len(moveTimes)
        avgSimDepths = sum(simDepths) / len(simDepths)
        avgRootSims = sum(rootSims) / len(rootSims)
        avgBoardEvals = sum(boardEvals) / len(boardEvals)
        avgNumWinMoveSets = sum(numWinMoveSets) / len(numWinMoveSets)

        writer.writerow(["avg", "-", avgNumMoves, f"{avgGameTimes:.1f}", avgMoveTimes,
            avgSimDepths, avgRootSims, avgBoardEvals, avgNumWinMoveSets
        ])


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
        mctsBotDefault2 = MonteCarloSearchTreeBot(
            numRootSimulations=MCTS_ITERS,
            maxSimDepth=MAX_DEPTH,
            evalFunc=evaluate,
            rememberPastBoardScores=False,
            conditionForSimulatingTriedMoves=lambda n: False,
            goForWin=False
        )
        run_matches(bot1=mctsBotDefault, bot2=mctsBotDefault2, numMatches=numPerBatch, fileName="results_default_self")

        mctsBotRemember2 = MonteCarloSearchTreeBot(
            numRootSimulations=MCTS_ITERS,
            maxSimDepth=MAX_DEPTH,
            evalFunc=evaluate,
            rememberPastBoardScores=True,
            conditionForSimulatingTriedMoves=lambda n: False,
            goForWin=False
        )
        run_matches(bot1=mctsBotRemember, bot2=mctsBotRemember2, numMatches=numPerBatch, fileName="results_remember_self")

        mctsBotSimTriedMoves2 = MonteCarloSearchTreeBot(
            numRootSimulations=MCTS_ITERS,
            maxSimDepth=MAX_DEPTH,
            evalFunc=evaluate,
            rememberPastBoardScores=False,
            conditionForSimulatingTriedMoves=None,
            goForWin=False
        )
        run_matches(bot1=mctsBotSimTriedMoves, bot2=mctsBotSimTriedMoves2, numMatches=numPerBatch, fileName="results_lookAtTriedMoves_self")
        
        mctsBotJumpTheWinGun2 = MonteCarloSearchTreeBot(
            numRootSimulations=MCTS_ITERS,
            maxSimDepth=MAX_DEPTH,
            evalFunc=evaluate,
            rememberPastBoardScores=False,
            conditionForSimulatingTriedMoves=lambda n: False,
            goForWin=True
        )
        run_matches(bot1=mctsBotJumpTheWinGun, bot2=mctsBotJumpTheWinGun2, numMatches=numPerBatch, fileName="results_jumpTheWinGun_self")

        mctsBotAllThree2 = MonteCarloSearchTreeBot(
            numRootSimulations=MCTS_ITERS,
            maxSimDepth=MAX_DEPTH,
            evalFunc=evaluate,
            rememberPastBoardScores=True,
            conditionForSimulatingTriedMoves=None,
            goForWin=True
        )
        run_matches(bot1=mctsBotAllThree, bot2=mctsBotAllThree2, numMatches=numPerBatch, fileName="results_allThree_self")
        return

    # against bot, is white
    run_matches(bot1=mctsBotDefault, bot2=botOpponent, numMatches=numPerBatch, fileName=f"results_default_asWhite_{botName}")
    run_matches(bot1=mctsBotRemember, bot2=botOpponent, numMatches=numPerBatch, fileName=f"results_remember_asWhite_{botName}")
    run_matches(bot1=mctsBotSimTriedMoves, bot2=botOpponent, numMatches=numPerBatch, fileName=f"results_lookAtTriedMoves_asWhite_{botName}")
    run_matches(bot1=mctsBotJumpTheWinGun, bot2=botOpponent, numMatches=numPerBatch, fileName=f"results_jumpTheWinGun_asWhite_{botName}")
    run_matches(bot1=mctsBotAllThree, bot2=botOpponent, numMatches=numPerBatch, fileName=f"results_allThree_asWhite_{botName}")

    # against bot, is black
    run_matches(bot2=mctsBotDefault, bot1=botOpponent, numMatches=numPerBatch, fileName=f"results_default_asBlack_{botName}")
    run_matches(bot2=mctsBotRemember, bot1=botOpponent, numMatches=numPerBatch, fileName=f"results_remember_asBlack_{botName}")
    run_matches(bot2=mctsBotSimTriedMoves, bot1=botOpponent, numMatches=numPerBatch, fileName=f"results_lookAtTriedMoves_asBlack_{botName}")
    run_matches(bot2=mctsBotJumpTheWinGun, bot1=botOpponent, numMatches=numPerBatch, fileName=f"results_jumpTheWinGun_asBlack_{botName}")
    run_matches(bot2=mctsBotAllThree, bot1=botOpponent, numMatches=numPerBatch, fileName=f"results_allThree_asBlack_{botName}")


if __name__ == "__main__":
    # run_matches(numMatches=10)
    runSeveralConfigdMatches(10)
    stockfish.configure({"UCI_Elo": STOCKFISH_ELO_DEFAULT})
    runSeveralConfigdMatches(10, botOpponent=stockfish, botName=f"stockfish-{STOCKFISH_ELO_DEFAULT}")
    stockfish.configure({"UCI_Elo": STOCKFISH_ELO_MIN})
    runSeveralConfigdMatches(10, botOpponent=stockfish, botName=f"stockfish-{STOCKFISH_ELO_MIN}")

    stockfish.quit()



