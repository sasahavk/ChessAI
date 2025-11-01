import time
import pygame, chess
import chess.engine
import asyncio

# colors
WHITE = pygame.Color(255, 255, 255)
BLACK = pygame.Color(0, 0, 0)
LIGHT_SQUARE = pygame.Color(240, 217, 181)
DARK_SQUARE = pygame.Color(181, 136, 99)
HIGHLIGHT = pygame.Color(255, 255, 0, 100)

WIDTH = 400
HEIGHT = 400
W = H =50


def get_square(x, y):
    f = x // W
    r = 7 - (y // H)
    return chess.square(f, r)


class ChessAI:
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2
        self.current_sqr = None
        self.has_selected = False
        self.highlighted_sqrs = []
        self.running = True
        self.font = None
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.board = chess.Board()
        self.engine = chess.engine.SimpleEngine.popen_uci(
            r"C:\Users\sasaa\OneDrive\Documents\GOLANG\src\MyVault\NOTES\UC-Davis\F25\ECS170\stockfish\stockfish-windows-x86-64-avx2.exe")

        self.board_array = [[None for _ in range(8)] for _ in range(8)]
        for square in range(64):
            piece = self.board.piece_at(square)
            if piece:
                self.board_array[7 - chess.square_rank(square)][chess.square_file(square)] = piece

    def draw_board(self):
        color = 0
        for i in range(8):
            t = 50 * i
            if i % 2 == 0:
                first_sq_color = LIGHT_SQUARE
            else:
                first_sq_color = DARK_SQUARE
            for j in range(8):
                l = 50 * j
                if j == 0:
                    color = first_sq_color
                else:
                    if color == LIGHT_SQUARE:
                        color = DARK_SQUARE
                    else:
                        color = LIGHT_SQUARE
                pygame.draw.rect(self.screen, color=color, rect=pygame.Rect(t, l, W, H))

    def draw_pieces(self):
        for i in range(8):
            for j in range(8):
                piece = self.board_array[i][j]
                if not piece:
                    continue
                color = WHITE
                if piece.color == chess.BLACK:
                    color = BLACK
                text_surface = self.font.render(piece.unicode_symbol(), True, color)
                text_rect = text_surface.get_rect(center=(W // 2 + W * j, H // 2 + H * i))
                self.screen.blit(text_surface, text_rect)

    def update_square(self, sqr):
        r = chess.square_rank(sqr)
        f = chess.square_file(sqr)
        color = None
        if (7 - r) % 2 == 0:
            if f % 2 == 0:
                color = LIGHT_SQUARE
            else:
                color = DARK_SQUARE
        else:
            if f % 2 == 0:
                color = DARK_SQUARE
            else:
                color = LIGHT_SQUARE

        pygame.draw.rect(self.screen, color=color, rect=pygame.Rect(50 * f, 50 * (7 - r), W, H))

    def update_piece(self, sqr):
        r = chess.square_rank(sqr)
        f = chess.square_file(sqr)
        piece = self.board.piece_at(sqr)

        if piece:
            if piece.color == chess.BLACK:
                color = BLACK
            else:
                color = WHITE
            text_surface = self.font.render(piece.unicode_symbol(), True, color)
            text_rect = text_surface.get_rect(center=(W // 2 + W * f, HEIGHT - (H // 2 + H * r)))
            self.screen.blit(text_surface, text_rect)

    def get_legal_moves(self, sqr):
        return [m for m in self.board.legal_moves if m.from_square == sqr]

    def highlight_legal_moves(self, mvs):
        highlighted = []
        for m in mvs:
            highlighted.append(m.to_square)
            r = chess.square_rank(m.to_square)
            f = chess.square_file(m.to_square)
            pygame.draw.rect(self.screen, color=HIGHLIGHT, rect=pygame.Rect(50 * f, 50 * (7 - r), W, H))
        return highlighted

    def remove_highlighted(self, sqrs):
        while sqrs:
            s = sqrs.pop(0)
            self.update_square(s)
            self.update_piece(s)

    def play(self):
        pygame.init()
        self.font = pygame.font.SysFont("segoeuisymbol", 36)
        clock = pygame.time.Clock()
        self.draw_board()
        self.draw_pieces()

        while self.running and not self.board.is_checkmate() and not self.board.is_stalemate():
            if self.board.turn == chess.WHITE:
                if self.p1 == "human":
                    self.play_human_move()
                else:
                    self.play_ai_move()
            else:
                if self.p2 == "human":
                    self.play_human_move()
                else:
                    self.play_ai_move()
            pygame.display.flip()
            clock.tick(60)  # limits FPS to 60

        pygame.quit()
        self.engine.quit()

    def play_human_move(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                selected_sqr = get_square(x, y)
                legal_moves = self.get_legal_moves(selected_sqr)

                if self.has_selected:
                    move = chess.Move(self.current_sqr, selected_sqr)
                    if move in self.board.legal_moves:
                        self.board.push(move)
                        self.update_square(self.current_sqr)
                        self.update_square(selected_sqr)
                        self.update_piece(selected_sqr)
                        self.has_selected = False
                        self.remove_highlighted(self.highlighted_sqrs)
                        self.highlighted_sqrs = []
                else:
                    self.highlighted_sqrs = self.highlight_legal_moves(legal_moves)
                    self.current_sqr = selected_sqr
                    self.has_selected = True

    def play_ai_move(self):
        time.sleep(1)
        current_move = self.engine.play(self.board, chess.engine.Limit(time=0.1)).move
        self.board.push(current_move)
        self.update_square(current_move.from_square)
        self.update_square(current_move.to_square)
        self.update_piece(current_move.to_square)

def main():
    game = ChessAI("human", "stockfish")
    game.play()

main()
