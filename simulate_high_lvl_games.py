import chess
import chess.engine
import random
import csv
from pathlib import Path
import numpy as np

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






