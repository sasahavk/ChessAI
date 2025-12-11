import random
import chess
import chess.engine
from ann_group.ann_eval import ann
from minimax_group import minimax_bot

class AnnGamesSim:
    def __init__(self, depth = 3):
        self.engine = chess.engine.SimpleEngine.popen_uci(
            r"C:\Users\sasaa\OneDrive\Documents\GOLANG\src\MyVault\NOTES\UC-Davis\F25\ECS170\stockfish\stockfish-windows-x86-64-avx2.exe")
        self.engine.configure({ "Skill Level": 1})
        self.ann = ann()
        self.ann_minimax = minimax_bot.MinimaxBot(depth=depth, eval_fn=self.ann.eval)

    def is_draw(self):
        return (self.board.ply() >= 16 and self.board.is_repetition(3)) or self.board.is_fifty_moves() or self.board.is_stalemate()

    def simulate_game(self):
        self.board = chess.Board()
        move = None
        has_draw = False

        while not self.board.is_game_over():
            if self.is_draw():
                if self.board.ply() >= 300:
                    print("LONG")
                has_draw = True
                break
            if self.board.turn == chess.WHITE:
                move = self.ann_minimax.play(self.board)
                print("WHITE: ", move)
            else:
                move = self.engine.play(self.board, chess.engine.Limit(time=0.1)).move
                print("BLACK: ", move)
            self.board.push(move)

        outcome = self.board.outcome()
        if has_draw:
            print("DRAW")
        elif outcome.winner == chess.WHITE:
            print("WHITE WINS")
        else:
            print("BLACK WINS")

sim = AnnGamesSim()
sim.simulate_game()