# main.py
import pygame
import chess
import chess.engine

from minimax_group.minimax_bot import MinimaxBot
from minimax_group.evaluate import evaluate
from mcts_group.mcts_bot import MonteCarloSearchTreeBot

import env_variables as env

TILE = 50
WIDTH = HEIGHT = TILE * 8

WHITE_RGB = (255, 255, 255)
BLACK_RGB = (0, 0, 0)
LIGHT_SQ = (240, 217, 181)
DARK_SQ = (181, 136, 99)
HILITE_RGBA = (255, 255, 0, 90)

STOCKFISH_LIMIT = chess.engine.Limit(time=0.1)  # or depth=12, nodes=...

# How long to display result screen (ms)
RESULT_DISPLAY_MS = 2500


def get_square_from_xy(x: int, y: int) -> chess.Square:
    """Convert screen (x,y) to chess.Square (0..63), with rank 7 at top row."""
    file_ = x // TILE
    rank_ = 7 - (y // TILE)
    return chess.square(file_, rank_)


class ChessGame:
    """
    Side strings:
      - "human": clicks
      - "minimax": our Python Minimax
      - "mcts": our Python Monte Carlo Tree Search
      - "stockfish": external engine (if STOCKFISH_PATH is set)
    """
    def __init__(self, white_player="human", black_player="minimax", minimax_depth=4, mcts_root_sim_count=10, mcts_depth=4):
        self.white_player = white_player
        self.black_player = black_player

        self.board = chess.Board()
        self.minimax = MinimaxBot(depth=minimax_depth, eval_fn=evaluate)
        self.mcts = MonteCarloSearchTreeBot(
            numRootSimulations=mcts_root_sim_count, maxSimDepth=mcts_depth, evalFunc=evaluate
        )

        self.screen = None
        self.font = None
        self.running = True

        self.has_selected = False
        self.current_sqr = None
        self.highlighted_sqrs = []

        self.highlight_layer = None

        self.engine = None
        if self.white_player == "stockfish" or self.black_player == "stockfish":
            if env.STOCKFISH_PATH:
                try:
                    self.engine = chess.engine.SimpleEngine.popen_uci(env.STOCKFISH_PATH)
                except Exception as e:
                    print(f"[WARN] Could not start Stockfish: {e}")
                    self.engine = None
            else:
                print("[INFO] STOCKFISH_PATH not set; Stockfish disabled for this run.")

    def draw_board(self):
        for r in range(8):
            for f in range(8):
                # flip rank index for color so a1 is dark
                color = DARK_SQ if (r + f) % 2 == 0 else LIGHT_SQ
                pygame.draw.rect(
                    self.screen,
                    color,
                    pygame.Rect(f * TILE, (7 - r) * TILE, TILE, TILE),
                )

    def draw_pieces_from_board(self):
        for sq in chess.SQUARES:
            piece = self.board.piece_at(sq)
            if not piece:
                continue
            color = WHITE_RGB if piece.color == chess.WHITE else BLACK_RGB
            r = chess.square_rank(sq)
            f = chess.square_file(sq)
            glyph = piece.unicode_symbol()
            text_surface = self.font.render(glyph, True, color)
            cx = TILE // 2 + TILE * f
            cy = TILE // 2 + TILE * (7 - r)
            rect = text_surface.get_rect(center=(cx, cy))
            self.screen.blit(text_surface, rect)

    def draw_highlights(self):
        self.highlight_layer.fill((0, 0, 0, 0))
        for sq in self.highlighted_sqrs:
            r = chess.square_rank(sq)
            f = chess.square_file(sq)
            pygame.draw.rect(
                self.highlight_layer,
                HILITE_RGBA,
                pygame.Rect(f * TILE, (7 - r) * TILE, TILE, TILE),
            )
        self.screen.blit(self.highlight_layer, (0, 0))

    def get_result_text(self) -> str:
        if self.board.is_checkmate():
            # side-to-move is checkmated
            winner = "Black" if self.board.turn == chess.WHITE else "White"
            return f"Checkmate — {winner} wins"
        # other draw types
        if self.board.is_stalemate():
            return "Draw — stalemate"
        outcome = self.board.outcome()
        if outcome and outcome.termination:
            if outcome.winner is None:
                return "Draw"
            return f"{'White' if outcome.winner else 'Black'} wins"
        return "Game over"

    def show_center_banner(self, text: str):
        """Render a centered banner text."""
        label = pygame.font.SysFont("arial", 28, bold=True).render(text, True, (30, 30, 30))
        rect = label.get_rect(center=(WIDTH // 2, HEIGHT // 2))
        # semi-transparent backdrop
        backdrop = pygame.Surface((rect.width + 20, rect.height + 12), pygame.SRCALPHA)
        backdrop.fill((255, 255, 255, 210))
        brect = backdrop.get_rect(center=(WIDTH // 2, HEIGHT // 2))
        self.screen.blit(backdrop, brect)
        self.screen.blit(label, rect)

    def choose_promotion(self, to_square: chess.Square, allowed_types: list[int]) -> int | None:
        """
        Modal mini-UI to choose promotion piece for the human.
        allowed_types is a list of piece types among {QUEEN, ROOK, BISHOP, KNIGHT}
        that are actually legal for the selected from→to.
        Returns the chosen piece type or None if canceled.
        """
        # Build dynamic choices from what's actually legal
        label_for = {
            chess.QUEEN:  "Q",
            chess.ROOK:   "R",
            chess.BISHOP: "B",
            chess.KNIGHT: "N",
        }
        priority = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]
        choices = [(pt, label_for[pt]) for pt in priority if pt in allowed_types]
        if not choices:
            return None

        f = chess.square_file(to_square)
        r = chess.square_rank(to_square)
        px = f * TILE
        py = (7 - r) * TILE
        item_w, item_h = 44, 44
        pad = 6
        popup_w = pad + len(choices) * (item_w + pad)
        popup_h = item_h + pad * 2

        popup_x = min(max(px - popup_w // 2 + TILE // 2, 4), WIDTH - popup_w - 4)
        popup_y = min(max(py - popup_h - 8, 4), HEIGHT - popup_h - 4)

        rects = []
        x = popup_x + pad
        for _ptype, _label in choices:
            rects.append(pygame.Rect(x, popup_y + pad, item_w, item_h))
            x += item_w + pad

        # Modal loop
        while True:
            # draw current board/background
            self.draw_board()
            self.draw_pieces_from_board()
            self.draw_highlights()

            # popup bg
            popup = pygame.Surface((popup_w, popup_h), pygame.SRCALPHA)
            popup.fill((40, 40, 40, 220))
            self.screen.blit(popup, (popup_x, popup_y))

            # draw items
            font = pygame.font.SysFont("arial", 22, bold=True)
            for idx, (ptype, label) in enumerate(choices):
                rct = rects[idx]
                pygame.draw.rect(self.screen, (220, 220, 220), rct, border_radius=6)
                glyph = font.render(label, True, (20, 20, 20))
                self.screen.blit(glyph, glyph.get_rect(center=rct.center))

            pygame.display.flip()

            # handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    return None
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    return None  # cancel
                if event.type == pygame.MOUSEBUTTONDOWN:
                    mx, my = event.pos
                    for idx, rct in enumerate(rects):
                        if rct.collidepoint(mx, my):
                            return choices[idx][0]  # piece type
                    # click outside popup cancels
                    if not pygame.Rect(popup_x, popup_y, popup_w, popup_h).collidepoint(mx, my):
                        return None

    # ---------- interaction ----------
    def handle_human_click(self, event) -> bool:
        """Process a click; return True iff a legal move was pushed."""
        if event.type != pygame.MOUSEBUTTONDOWN:
            return False
        x, y = event.pos
        target_sqr = get_square_from_xy(x, y)

        if not self.has_selected:
            piece = self.board.piece_at(target_sqr)
            # only select your own piece
            if piece and piece.color == self.board.turn:
                self.current_sqr = target_sqr
                # legal targets from that square
                self.highlighted_sqrs = [m.to_square for m in self.board.legal_moves
                                         if m.from_square == target_sqr]
                self.has_selected = True
            else:
                self.current_sqr = None
                self.highlighted_sqrs = []
            return False

        # ---- Second click: attempt a move ----
        from_sq = self.current_sqr
        to_sq = target_sqr

        # Quick deselect: clicking the same square toggles off
        if to_sq == from_sq:
            self.has_selected = False
            self.current_sqr = None
            self.highlighted_sqrs = []
            return False

        legal_moves = list(self.board.legal_moves)

        # All legal promotions for exactly this from→to
        promotion_moves = [
            m for m in legal_moves
            if m.from_square == from_sq and m.to_square == to_sq and m.promotion
        ]

        move_to_push = None

        if promotion_moves:
            # Restrict UI to actually legal promotion piece types
            allowed_types = sorted({m.promotion for m in promotion_moves})
            chosen = self.choose_promotion(to_sq, allowed_types)
            if chosen is None:
                # canceled; keep selection so user can re-try
                return False

            candidate = chess.Move(from_sq, to_sq, promotion=chosen)
            if candidate in legal_moves:
                move_to_push = candidate
            else:
                move_to_push = promotion_moves[0]
        else:
            # Non-promotion path
            candidate = chess.Move(from_sq, to_sq)
            if candidate in legal_moves:
                move_to_push = candidate

        made_move = False
        if move_to_push is not None:
            self.board.push(move_to_push)
            made_move = True

        # reset selection
        self.has_selected = False
        self.current_sqr = None
        self.highlighted_sqrs = []
        return made_move

    # ---------- AI turns ----------
    def play_minimax_turn(self):
        mv = self.minimax.play(self.board)
        if mv:
            self.board.push(mv)
    
    def play_mcts_turn(self):
        move = self.mcts.play(self.board)
        if move:
            self.board.push(move)
        else:
            print("uh oh there should be a mcts move")

    def play_stockfish_turn(self):
        if not self.engine:
            self.play_minimax_turn()
            return
        try:
            result = self.engine.play(self.board, STOCKFISH_LIMIT)
            if result and result.move:
                self.board.push(result.move)
        except Exception as e:
            print(f"[WARN] Stockfish error: {e}")
            # graceful fallback
            self.play_minimax_turn()

    # ---------- main loop ----------
    def play(self):
        pygame.init()
        pygame.display.set_caption("Chess — Human / Minimax / Monte Carlo / Stockfish")
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.font = pygame.font.SysFont(env.FONT_NAME, env.FONT_SIZE)
        self.highlight_layer = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        clock = pygame.time.Clock()

        while self.running and not self.board.is_game_over():
            moved_this_frame = False

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    self.engine.quit()
                else:
                    if self.board.turn == chess.WHITE and self.white_player == "human":
                        moved_this_frame |= self.handle_human_click(event)
                    elif self.board.turn == chess.BLACK and self.black_player == "human":
                        moved_this_frame |= self.handle_human_click(event)

            # draw current state
            self.draw_board()
            self.draw_pieces_from_board()
            self.draw_highlights()
            pygame.display.flip()

            if moved_this_frame:
                clock.tick(60)
                continue

            # 2) AI turn
            if self.running:
                if self.board.turn == chess.WHITE and self.white_player != "human":
                    if self.white_player == "minimax":
                        self.play_minimax_turn()
                    elif self.white_player == "mcts":
                        self.play_mcts_turn()
                    elif self.white_player == "stockfish":
                        self.play_stockfish_turn()
                elif self.board.turn == chess.BLACK and self.black_player != "human":
                    if self.black_player == "minimax":
                        self.play_minimax_turn()
                    elif self.black_player == "mcts":
                        self.play_mcts_turn()
                    elif self.black_player == "stockfish":
                        self.play_stockfish_turn()

            self.draw_board()
            self.draw_pieces_from_board()
            self.draw_highlights()
            pygame.display.flip()
            clock.tick(60)

        # ----- Result overlay for a few seconds, then close -----
        if self.running:
            self.draw_board()
            self.draw_pieces_from_board()
            self.draw_highlights()
            self.show_center_banner(self.get_result_text())
            pygame.display.flip()

            # Wait until user manually closes the window
            self.engine.quit()
            waiting = True
            while waiting:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        waiting = False
                pygame.time.Clock().tick(30)

def main():
    # Choose players per side: "human", "minimax", or "stockfish"
    # Example: Minimax (white) vs Human (black)
    game = ChessGame(white_player="human", black_player="mcts", minimax_depth=4)
    game.play()


if __name__ == "__main__":
    main()
