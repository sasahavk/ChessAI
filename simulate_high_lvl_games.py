import chess
import chess.engine
import random
import csv
from pathlib import Path

skill_lvl_rng = [16,20]
threads_all = [1,2,4]
hash_all = [16, 32, 64]
UCI_Elo_rng= [2500, 3190]


def init_engine():
    engine = chess.engine.SimpleEngine.popen_uci(
        r"C:\Users\sasaa\OneDrive\Documents\GOLANG\src\MyVault\NOTES\UC-Davis\F25\ECS170\stockfish\stockfish-windows-x86-64-avx2.exe")

    skill = random.randint(skill_lvl_rng[0], skill_lvl_rng[1])
    threads = random.choice(threads_all)
    hash = random.choice(hash_all)
    UCI_Elo = random.randint(UCI_Elo_rng[0], UCI_Elo_rng[1])
    engine.configure(
        {"Hash": hash, "Threads": threads, "Skill Level": skill,
         "UCI_Elo": UCI_Elo})
    return engine


def add_game_outcome(winner, game_log):
    for i in range(len(game_log)):
        if winner == game_log[i][1]:
            game_log[i][3] =  2
        elif winner == None:
            game_log[i][3] = 1
        else:
            game_log[i][3] = 0
    return game_log


def board_to_string(board):
    board_str = ''.join(board.piece_at(i).symbol() if board.piece_at(i) else '.' for i in range(64))
    return board_str


def simulate_game():
    board = chess.Board()
    engine_WHITE = init_engine()
    engine_BLACK = init_engine()
    move = None
    game_log = []

    while not board.is_game_over():
        move_no =  (board.ply() //2)+1
        game_log.append([board_to_string(board), board.turn,move_no, 0])
        if board.turn == chess.WHITE:
            move = engine_WHITE.play(board, chess.engine.Limit(time=0.1)).move
        else:
            move = engine_BLACK.play(board, chess.engine.Limit(time=0.1)).move
        board.push(move)

    engine_WHITE.quit()
    engine_BLACK.quit()
    return add_game_outcome(board.outcome().winner, game_log)


def init_file(file_name):
    path = Path(file_name)
    if not path.exists():
        with path.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["board", "turn", "move_num", "result"])
    return path


def generate_positions(file_name, game_count):
    path = init_file(file_name)

    for i in range(game_count):
        with path.open("a", newline="", encoding="utf-8") as f:
            game_positions = simulate_game()
            csv.writer(f).writerows(game_positions)
        print(i)

    print("DONE")


generate_positions("games_high.csv", 411)




# CenterControl
    #cc[] = {12,12,18,24}
# bishop pair
    # if (white_bishops >= 2) score += 45;
    # if (black_bishops >= 2) score -= 45;
# Outposts (Knight/Bishop) (+30)
    # if (knight on e5/d5 && supported by own pawn) score += 35;
    # if (bishop on d6/e6 && supported) score += 28;
# Passed Pawns (Beyond Structure) (+40 Elo)
    #if (is_passed(pawn)) score += 80 / distance_to_promotion;

# piece square tables
    # pawn
        #  0,  0,  0,  0,  0,  0,  0,  0,
        # 50, 50, 50, 50, 50, 50, 50, 50,
        # 10, 10, 20, 30, 30, 20, 10, 10,
        #  5,  5, 10, 25, 25, 10,  5,  5,
        #  0,  0,  0, 20, 20,  0,  0,  0,
        #  5, -5,-10,  0,  0,-10, -5,  5,
        #  5, 10, 10,-20,-20, 10, 10,  5,
        #  0,  0,  0,  0,  0,  0,  0,  0
    # knight
        # -50,-40,-30,-30,-30,-30,-40,-50,
        # -40,-20,  0,  0,  0,  0,-20,-40,
        # -30,  0, 10, 15, 15, 10,  0,-30,
        # -30,  5, 15, 20, 20, 15,  5,-30,
        # -30,  0, 15, 20, 20, 15,  0,-30,
        # -30,  5, 10, 15, 15, 10,  5,-30,
        # -40,-20,  0,  5,  5,  0,-20,-40,
        # -50,-40,-30,-30,-30,-30,-40,-50,
    # bishop
        # -20,-10,-10,-10,-10,-10,-10,-20,
        # -10,  0,  0,  0,  0,  0,  0,-10,
        # -10,  0,  5, 10, 10,  5,  0,-10,
        # -10,  5,  5, 10, 10,  5,  5,-10,
        # -10,  0, 10, 10, 10, 10,  0,-10,
        # -10, 10, 10, 10, 10, 10, 10,-10,
        # -10,  5,  0,  0,  0,  0,  5,-10,
        # -20,-10,-10,-10,-10,-10,-10,-20,
    # rook
        #   0,  0,  0,  0,  0,  0,  0,  0,
        #   5, 10, 10, 10, 10, 10, 10,  5,
        #  -5,  0,  0,  0,  0,  0,  0, -5,
        #  -5,  0,  0,  0,  0,  0,  0, -5,
        #  -5,  0,  0,  0,  0,  0,  0, -5,
        #  -5,  0,  0,  0,  0,  0,  0, -5,
        #  -5,  0,  0,  0,  0,  0,  0, -5,
        #   0,  0,  0,  5,  5,  0,  0,  0
    # queen
        # -20,-10,-10, -5, -5,-10,-10,-20,
        # -10,  0,  0,  0,  0,  0,  0,-10,
        # -10,  0,  5,  5,  5,  5,  0,-10,
        #  -5,  0,  5,  5,  5,  5,  0, -5,
        #   0,  0,  5,  5,  5,  5,  0, -5,
        # -10,  5,  5,  5,  5,  5,  0,-10,
        # -10,  0,  5,  0,  0,  0,  0,-10,
        # -20,-10,-10, -5, -5,-10,-10,-20
    # king middle game
        # -30,-40,-40,-50,-50,-40,-40,-30,
        # -30,-40,-40,-50,-50,-40,-40,-30,
        # -30,-40,-40,-50,-50,-40,-40,-30,
        # -30,-40,-40,-50,-50,-40,-40,-30,
        # -20,-30,-30,-40,-40,-30,-30,-20,
        # -10,-20,-20,-20,-20,-20,-20,-10,
        #  20, 20,  0,  0,  0,  0, 20, 20,
        #  20, 30, 10,  0,  0, 10, 30, 20
    # king end game
        # -50,-40,-30,-20,-20,-30,-40,-50,
        # -30,-20,-10,  0,  0,-10,-20,-30,
        # -30,-10, 20, 30, 30, 20,-10,-30,
        # -30,-10, 30, 40, 40, 30,-10,-30,
        # -30,-10, 30, 40, 40, 30,-10,-30,
        # -30,-10, 20, 30, 30, 20,-10,-30,
        # -30,-30,  0,  0,  0,  0,-30,-30,
        # -50,-30,-30,-30,-30,-30,-30,-50