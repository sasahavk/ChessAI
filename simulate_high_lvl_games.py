import chess
import chess.engine
import random
import csv
from pathlib import Path
import numpy as np

skill_lvl_rng = [1,20]
threads_all = [1,2,4]
hash_all = [16, 32, 64]
UCI_Elo_rng= [1320, 3190]


class ChessSimulator:
    def __init__(self, batch_count: int, batch_size: int, file_name, engine_count:int):
        self.engine_count = engine_count
        self.batch_count = batch_count
        self.batch_size = batch_size
        self.file_name = file_name

        self.engines = [None]*self.engine_count
        self.init_engines()

        self.MAX_MOVES = 150
        self.file_path = None
        self.init_file()

        self.board = None
        self.first_position_idx = 0
        self.position_idx = 0
        self.game_positions = [[0, 0, 0, 0] for _ in range(self.MAX_MOVES*self.batch_size)]

    def init_engines(self):
        for i in range(self.engine_count):
            self.engines[i] = chess.engine.SimpleEngine.popen_uci(
            r"C:\Users\sasaa\OneDrive\Documents\GOLANG\src\MyVault\NOTES\UC-Davis\F25\ECS170\stockfish\stockfish-windows-x86-64-avx2.exe")
            self.engines[i].configure({
                "Hash": random.choice(hash_all),
                "Threads": random.choice(threads_all),
                "Skill Level": random.randint(skill_lvl_rng[0], skill_lvl_rng[1]),
                "UCI_Elo": random.randint(UCI_Elo_rng[0], UCI_Elo_rng[1])
            })

    def init_file(self):
        path = Path(self.file_name)
        if not path.exists():
            with path.open("w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(["board", "turn", "move_num", "result"])
        self.file_path = path

    def add_game_outcome(self, has_draw: bool):
        outcome = self.board.outcome()
        has_printed  = False
        for i in range(self.position_idx - self.first_position_idx):
            if has_draw:
                if not has_printed:
                    has_printed = True
                self.game_positions[self.first_position_idx + i][3] = 1
            elif not outcome:
                if not has_printed:
                    has_printed = True
                self.game_positions[self.first_position_idx + i][3] = 1
            elif outcome.winner == self.game_positions[self.first_position_idx + i][1]:
                self.game_positions[self.first_position_idx + i][3] = 2
            else:
                self.game_positions[self.first_position_idx + i][3] = 0

    def board_to_string(self):
        board_str = ''.join(self.board.piece_at(i).symbol() if self.board.piece_at(i) else '.' for i in range(64))
        return board_str

    def simulate_game(self):
        self.board = chess.Board()
        move = None
        white_engine_id = random.randint(0,self.engine_count-1)
        black_engine_id = random.randint(0,self.engine_count-1)

        has_draw = False

        while not self.board.is_game_over():
            if self.is_draw():
                has_draw = True
                break
            move_no = (self.board.ply() //2)+1
            self.game_positions[self.position_idx] = [self.board.fen(), self.board.turn, move_no, 0]

            if self.board.turn == chess.WHITE:
                move = self.engines[white_engine_id].play(self.board, chess.engine.Limit(time=0.1)).move
            else:
                move = self.engines[black_engine_id].play(self.board, chess.engine.Limit(time=0.1)).move

            self.board.push(move)
            self.position_idx += 1

        self.add_game_outcome(has_draw)
        self.first_position_idx = self.position_idx

    def quit_engines(self):
        for i in range(self.engine_count):
            self.engines[i].quit()

    def is_draw(self):
        return (self.board.ply() >= 16 and self.board.is_repetition(3)) or self.board.is_fifty_moves() or self.board.is_stalemate() or self.board.ply() >= 140

    def generate_positions(self):
        for i in range(self.batch_count):
            for j in range(self.batch_size):
                self.simulate_game()
                print(i, j, self.first_position_idx)

            with self.file_path.open("a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerows(self.game_positions[:self.position_idx])
            self.game_positions = [[0, 0, 0, 0] for _ in range(self.MAX_MOVES*self.batch_size)]
            self.first_position_idx = 0
            self.position_idx = 0

        self.quit_engines()


simulator = ChessSimulator(10,50, "games_new.csv", 40)
simulator.generate_positions()




