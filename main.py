# main.py
import pygame
import chess
import chess.engine
import csv
from pathlib import Path
import os
import shutil
import time
from minimax_group.minimax_bot import MinimaxBot
from minimax_group.evaluate import evaluate
from minimax_group.minimax_new import FastMinimaxBot
import env_variables as env
from chess.engine import SimpleEngine, Info

TILE = 50
WIDTH = HEIGHT = TILE * 8

WHITE_RGB = (255, 255, 255)
BLACK_RGB = (0, 0, 0)
LIGHT_SQ = (185, 160, 130)
DARK_SQ  = (125, 85, 45)

#LIGHT_SQ = (240, 217, 181)
#DARK_SQ = (181, 136, 99)
HILITE_RGBA = (255, 255, 0, 90)


# Stockfish: set path when you’re ready
STOCKFISH_LIMIT = chess.engine.Limit(time=2.0)  # or depth=12, nodes=...
STOCKFISH_ELO = 1900

# How long to display result screen (ms)
RESULT_DISPLAY_MS = 2500

PIECE_GLYPHS = {
    chess.PAWN:   ("P", "p"),  # white, black
    chess.KNIGHT: ("N", "n"),
    chess.BISHOP: ("B", "b"),
    chess.ROOK:   ("R", "r"),
    chess.QUEEN:  ("Q", "q"),
    chess.KING:   ("K", "k"),
}


def load_stockfish():
    # 1. Try system-installed Stockfish
    system_path = shutil.which("stockfish")
    if system_path:
        print("[INFO] Using system Stockfish:", system_path)
        try:
            return SimpleEngine.popen_uci(system_path)
        except Exception as e:
            print("[WARN] Failed to load system Stockfish:", e)

    # 2. Fall back to bundled binary inside repo
    root = os.path.dirname(os.path.abspath(__file__))
    local_path = os.path.join(root, "stockfish", "stockfish","stockfish-windows-x86-64-avx2.exe")

    if os.path.exists(local_path):
        print("[INFO] Using bundled Stockfish:", local_path)
        try:
            return SimpleEngine.popen_uci(local_path)
        except Exception as e:
            print("[ERROR] Failed to launch bundled Stockfish:", e)

    # 3. If everything failed
    print("[ERROR] No Stockfish available. Minimax-only mode.")
    return None



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
      - "stockfish": external engine (if env.STOCKFISH_PATH is set)
    """
    def __init__(self, white_player="human", black_player="minimax", minimax_depth=4, flip_board=False):
        self.white_player = white_player
        self.black_player = black_player

        self.board = chess.Board()
        self.minimax = MinimaxBot(depth=minimax_depth, eval_fn=evaluate)
        self.minimax_new = FastMinimaxBot(depth=minimax_depth, eval_fn=evaluate)
        self.flip_board = flip_board
        self.screen = None
        self.font = None
        self.running = True

        self.has_selected = False
        self.current_sqr = None
        self.highlighted_sqrs = []

        self.highlight_layer = None
        self.last_move: chess.Move | None = None
        self.last_move_squares: list[chess.Square] = []

        self.minimax_time_total = 0.0
        self.minimax_moves = 0
        self.minimax_new_time_total = 0.0
        self.minimax_new_moves = 0

        self.move_log: list[dict] = []

        self.stockfish_time_total = 0.0
        self.stockfish_moves = 0

        self.engine = None
        if self.white_player == "stockfish" or self.black_player == "stockfish":
            self.engine = load_stockfish()

        if self.engine:
            try:
                # example: set strength to ~1500 Elo
                self.engine.configure({"UCI_LimitStrength": True, "UCI_Elo": STOCKFISH_ELO})
            except Exception as e:
                print(f"[WARN] Could not configure Stockfish options: {e}")

        self.engine_analyze = None
        if self.white_player == "stockfish" or self.black_player == "stockfish":
            self.engine_analyze = load_stockfish()

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
        if self.last_move_squares:
            for sq in self.last_move_squares:
                r = chess.square_rank(sq)
                f = chess.square_file(sq)
                pygame.draw.rect(
                    self.highlight_layer,
                    (255, 220, 120, 110),  # orange-ish, semi-transparent
                    pygame.Rect(f * TILE, (7 - r) * TILE, TILE, TILE),
                )

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
            san_str = self.board.san(move_to_push)
            self.board.push(move_to_push)
            print("Human played:", san_str)
            made_move = True
            self.last_move = move_to_push
            self.last_move_squares = [move_to_push.from_square, move_to_push.to_square]

        # reset selection
        self.has_selected = False
        self.current_sqr = None
        self.highlighted_sqrs = []
        return made_move

    def _sf_analyse(self, board: chess.Board, pov_color: chess.Color | None = None) -> dict | None:
        if not self.engine_analyze:
            return None

        # Ask python-chess for all available info (new API uses Info flags, not strings)
        info = self.engine_analyze.analyse(
            board,
            STOCKFISH_LIMIT,
            info=Info.ALL,
        )

        eval_cp = None
        score = info.get("score")
        if score is not None:
            if pov_color is None:
                pov_color = board.turn
            pov = score.pov(pov_color)
            if pov.is_mate():
                mate_in = pov.mate()
                if mate_in is not None:
                    sign = 1 if mate_in > 0 else -1
                    eval_cp = sign * 100000
            else:
                eval_cp = pov.score()

        pv = info.get("pv")
        best_move = pv[0] if pv else None

        return {
            "eval_cp": eval_cp,
            "depth": info.get("depth"),
            "nodes": info.get("nodes"),
            "nps": info.get("nps"),
            "time": info.get("time"),
            "best_move": best_move,
        }

    def _extract_eval_cp(self, info: dict, pov_color: chess.Color) -> int | None:
        score = info.get("score")
        if score is None:
            return None

        pov = score.pov(pov_color)

        if pov.is_mate():
            mate_in = pov.mate()
            if mate_in is not None:
                sign = 1 if mate_in > 0 else -1
                return sign * 100000
            return None
        else:
            return pov.score()

    def log_minimax_move(self, move: chess.Move, move_san: str, color: chess.Color,
                         elapsed: float,
                         pre_info: dict | None,
                         post_info: dict | None):
        """Log one Minimax move + Stockfish analysis into move_log."""
        ply = len(self.board.move_stack)

        sf_eval_cp = post_info["eval_cp"] if post_info else None
        sf_depth = pre_info["depth"] if pre_info else None
        sf_nodes = pre_info["nodes"] if pre_info else None
        sf_nps = pre_info["nps"] if pre_info else None
        sf_time = pre_info["time"] if pre_info else None

        # centipawn loss: from Minimax side's POV
        cp_loss = None
        error_type = None
        sf_agreement = None

        if pre_info and post_info:
            pre_eval = pre_info["eval_cp"]
            post_eval = post_info["eval_cp"]
            if pre_eval is not None and post_eval is not None:
                cp_loss = pre_eval - post_eval  # >0 means position got worse for Minimax
                delta = max(0, cp_loss)
                if delta <= 50:
                    error_type = "ok"
                elif delta <= 150:
                    error_type = "inaccuracy"
                elif delta <= 300:
                    error_type = "mistake"
                else:
                    error_type = "blunder"

            best_move = pre_info.get("best_move")
            if best_move is not None:
                sf_agreement = 1 if best_move == move else 0
            else:
                sf_agreement = None

        # Minimax search stats (from MinimaxBot)
        mb = self.minimax
        nodes = getattr(mb, "nodes", None)
        qs_nodes = getattr(mb, "qs_nodes", None)
        depth_reached = getattr(mb, "search_depth_reached", None)
        alpha_beta_cutoffs = getattr(mb, "alpha_beta_cutoffs", None)
        null_move_attempts = getattr(mb, "null_move_attempts", None)
        null_move_cutoffs = getattr(mb, "null_move_cutoffs", None)
        see_prunes = getattr(mb, "see_prunes", None)
        tt_hits = getattr(mb, "tt_hits", None)
        tt_stores = getattr(mb, "tt_stores", None)

        nps_minimax = None
        if elapsed > 0 and nodes is not None:
            nps_minimax = nodes / elapsed

        self.move_log.append({
            "ply": ply,
            "side": "white" if color == chess.WHITE else "black",
            "engine": "minimax",
            "move_uci": move.uci(),
            "move_san": move_san,

            # Move Quality vs Stockfish
            "sf_eval_cp": sf_eval_cp,
            "sf_depth": sf_depth,
            "sf_nodes": sf_nodes,
            "sf_nps": sf_nps,
            "sf_time": sf_time,

            "cp_loss": cp_loss,
            "error_type": error_type,
            "sf_agreement": sf_agreement,

            # Search stats (Minimax only)
            "nodes_searched": nodes,
            "qs_nodes": qs_nodes,
            "search_depth": depth_reached,
            "time_spent": elapsed,
            "nps_minimax": nps_minimax,

            # Pruning stats
            "alpha_beta_cutoffs": alpha_beta_cutoffs,
            "null_move_attempts": null_move_attempts,
            "null_move_cutoffs": null_move_cutoffs,
            "see_prunes": see_prunes,
            "tt_hits": tt_hits,
            "tt_stores": tt_stores,
        })

    def log_stockfish_move(self, move: chess.Move, move_san: str, color: chess.Color,
                           elapsed: float,
                           info: dict | None):
        """Log one Stockfish move into move_log (minimal fields)."""
        ply = len(self.board.move_stack)

        sf_eval_cp = info["eval_cp"] if info else None
        sf_depth = info["depth"] if info else None
        sf_nodes = info["nodes"] if info else None
        sf_nps = info["nps"] if info else None
        sf_time = info["time"] if info else None

        self.move_log.append({
            "ply": ply,
            "side": "white" if color == chess.WHITE else "black",
            "engine": "stockfish",
            "move_uci": move.uci(),
            "move_san": move_san,

            # Stockfish metrics
            "sf_eval_cp": sf_eval_cp,
            "sf_depth": sf_depth,
            "sf_nodes": sf_nodes,
            "sf_nps": sf_nps,
            "sf_time": sf_time,

            # No quality or search stats for Stockfish (Minimax-only fields)
            "cp_loss": None,
            "error_type": None,
            "sf_agreement": None,

            "nodes_searched": None,
            "qs_nodes": None,
            "search_depth": None,
            "time_spent": elapsed,   # we *do* log total time per SF move
            "nps_minimax": None,

            "alpha_beta_cutoffs": None,
            "null_move_attempts": None,
            "null_move_cutoffs": None,
            "see_prunes": None,
            "tt_hits": None,
            "tt_stores": None,
        })


    # ---------- AI turns ----------
    def play_minimax_turn(self):
        color_to_move = self.board.turn  # who is about to move (for logging)

        # 1) Stockfish eval BEFORE Minimax move (pre_info)
        pre_info = self._sf_analyse(self.board, pov_color=color_to_move) if self.engine_analyze else None

        # 2) Let Minimax search
        start = time.perf_counter()
        mv = self.minimax.play(self.board)
        elapsed = time.perf_counter() - start

        if mv:
            san_str = self.board.san(mv)
            self.board.push(mv)
            print(f"Minimax played: {san_str}  (t = {elapsed:.3f}s)")
            self.last_move = mv
            self.last_move_squares = [mv.from_square, mv.to_square]

            # 3) Stockfish eval AFTER Minimax move (post_info)
            post_info = self._sf_analyse(self.board, pov_color=color_to_move) if self.engine_analyze else None

            # 4) Update per-game stats
            self.minimax_time_total += elapsed
            self.minimax_moves += 1

            # 5) Log full Minimax + Stockfish-analysis data
            self.log_minimax_move(
                move=mv,
                move_san=san_str,
                color=color_to_move,
                elapsed=elapsed,
                pre_info=pre_info,
                post_info=post_info,
            )

    def play_minimax_new_turn(self):
        start = time.perf_counter()
        mv = self.minimax_new.play(self.board)
        elapsed = time.perf_counter() - start

        if mv:
            san_str = self.board.san(mv)
            self.board.push(mv)
            print(f"NewMinimax played: {san_str}  (t = {elapsed:.3f}s)")
            self.last_move = mv
            self.last_move_squares = [mv.from_square, mv.to_square]

            # update stats
            self.minimax_new_time_total += elapsed
            self.minimax_new_moves += 1

    def play_stockfish_turn(self):
        if not self.engine:
            self.play_minimax_turn()
            return

        color_to_move = self.board.turn

        # Single search: get move + stats in one call
        start = time.perf_counter()
        result = self.engine.play(
            self.board,
            STOCKFISH_LIMIT,
            info=Info.ALL,
        )
        elapsed = time.perf_counter() - start

        mv = result.move
        sf_info = result.info

        if mv is None:
            print("[WARN] Stockfish play() did not return a move")
            return

        san_str = self.board.san(mv)
        self.board.push(mv)
        print(f"Stockfish played: {san_str}  (t = {elapsed:.3f}s)")
        self.last_move = mv
        self.last_move_squares = [mv.from_square, mv.to_square]

        # Update timing stats
        self.stockfish_time_total += elapsed
        self.stockfish_moves += 1

        # Log using the same info dict
        self.log_stockfish_move(
            move=mv,
            move_san=san_str,
            color=color_to_move,
            elapsed=elapsed,
            info={
                "eval_cp": self._extract_eval_cp(sf_info, pov_color=color_to_move),
                "depth": sf_info.get("depth"),
                "nodes": sf_info.get("nodes"),
                "nps": sf_info.get("nps"),
                "time": sf_info.get("time"),
                "best_move": mv,  # PV[0] is effectively this move
            },
        )

    # ---------- main loop ----------
    # In ChessGame
    def play(self, render: bool | None = None, block_on_gameover: bool | None = None):
        game_start_time = time.perf_counter()
        # Auto settings: show UI if a human is involved; block at end iff we are rendering
        if render is None:
            render = (self.white_player == "human" or self.black_player == "human")
        if block_on_gameover is None:
            block_on_gameover = render

        if render:
            pygame.init()
            pygame.display.set_caption("Chess — Human / Minimax / Stockfish")
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            self.font = pygame.font.SysFont(env.FONT_NAME, env.FONT_SIZE)

            self.highlight_layer = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            clock = pygame.time.Clock()
        else:
            clock = None  # headless

        # --- main loop unchanged, but guard all drawing/event code with `if render:` ---

        while self.running and not self.board.is_game_over():
            moved_this_frame = False

            if render:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False
                    else:
                        if self.board.turn == chess.WHITE and self.white_player == "human":
                            moved_this_frame |= self.handle_human_click(event)
                        elif self.board.turn == chess.BLACK and self.black_player == "human":
                            moved_this_frame |= self.handle_human_click(event)

                self.draw_board()
                self.draw_highlights()
                self.draw_pieces_from_board()
                #self.draw_highlights()
                pygame.display.flip()

                if moved_this_frame:
                    clock.tick(60)
                    continue

            # AI turn(s)
            if self.running:
                if self.board.turn == chess.WHITE and self.white_player != "human":
                    if self.white_player == "minimax":
                        self.play_minimax_turn()
                    elif self.white_player == "new_minimax":
                        self.play_minimax_new_turn()
                    elif self.white_player == "stockfish":
                        self.play_stockfish_turn()
                elif self.board.turn == chess.BLACK and self.black_player != "human":
                    if self.black_player == "minimax":
                        self.play_minimax_turn()
                    elif self.black_player == "new_minimax":
                        self.play_minimax_new_turn()
                    elif self.black_player == "stockfish":
                        self.play_stockfish_turn()

            if render:
                self.draw_board()
                self.draw_highlights()
                self.draw_pieces_from_board()
                #self.draw_highlights()
                pygame.display.flip()
                clock.tick(60)

        # ----- Game over -----
        # Decide winner/label once
        # ----- Game over -----
        outcome = self.board.outcome()
        if outcome and outcome.winner is not None:
            winner = "white" if outcome.winner else "black"
        elif self.board.is_checkmate():
            winner = "black" if self.board.turn == chess.WHITE else "white"
        else:
            winner = "draw"

        # Termination type
        if outcome and outcome.termination:
            termination = outcome.termination.name.lower()  # e.g. "checkmate", "stalemate", ...
        elif self.board.is_checkmate():
            termination = "checkmate"
        elif self.board.is_stalemate():
            termination = "stalemate"
        else:
            termination = "unknown"

        game_length_plies = len(self.board.move_stack)
        total_game_time = time.perf_counter() - game_start_time

        avg_time_minimax = (self.minimax_time_total / self.minimax_moves) if self.minimax_moves > 0 else 0.0
        avg_time_stockfish = (self.stockfish_time_total / self.stockfish_moves) if self.stockfish_moves > 0 else 0.0

        print(f"[RESULT] Winner: {winner}  (white={self.white_player}, black={self.black_player})")
        if self.minimax_moves > 0:
            print(f"[STATS] Minimax: {self.minimax_moves} moves, avg {avg_time_minimax:.3f} s/move")
        if self.stockfish_moves > 0:
            print(f"[STATS] Stockfish: {self.stockfish_moves} moves, avg {avg_time_stockfish:.3f} s/move")

        # --- Per-game CSV (A) ---
        results_path = Path("results_log.csv")
        write_header = not results_path.exists()

        # Compute game_id from existing rows
        if write_header:
            existing_games = 0
        else:
            with results_path.open("r", encoding="utf-8") as rf:
                existing_games = sum(1 for _ in rf) - 1  # minus header
        game_id = existing_games + 1

        with results_path.open("a", newline="", encoding="utf-8") as f:
            fieldnames = [
                "game_id",
                "winner",
                "termination",
                "white_player",
                "black_player",
                "game_length_plies",
                "avg_time_per_move_minimax",
                "avg_time_per_move_stockfish",
                "total_game_time",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()

            writer.writerow({
                "game_id": game_id,
                "winner": winner,
                "termination": termination,
                "white_player": self.white_player,
                "black_player": self.black_player,
                "game_length_plies": game_length_plies,
                "avg_time_per_move_minimax": avg_time_minimax,
                "avg_time_per_move_stockfish": avg_time_stockfish,
                "total_game_time": total_game_time,
            })

        # --- Per-move CSV (B + Stockfish moves) ---
        move_stats_path = Path("move_stats.csv")
        write_header_moves = not move_stats_path.exists()

        with move_stats_path.open("a", newline="", encoding="utf-8") as f:
            fieldnames = [
                "game_id",
                "ply",
                "side",
                "engine",
                "move_san",
                "move_uci",
                "sf_eval_cp",
                "sf_depth",
                "sf_nodes",
                "sf_nps",
                "sf_time",
                "cp_loss",
                "error_type",
                "sf_agreement",
                "nodes_searched",
                "qs_nodes",
                "search_depth",
                "time_spent",
                "nps_minimax",
                "alpha_beta_cutoffs",
                "null_move_attempts",
                "null_move_cutoffs",
                "see_prunes",
                "tt_hits",
                "tt_stores",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header_moves:
                writer.writeheader()

            for row in self.move_log:
                out = row.copy()
                out["game_id"] = game_id
                writer.writerow(out)


def run_batch(num_games=10):
    for i in range(1, num_games + 1):
        print(f"\n=== Starting game {i} ===")

        game = ChessGame(
            white_player="minimax",
            black_player="stockfish",
            minimax_depth=5,
            flip_board=False,
        )

        start_time = time.time()
        is_last = (i == num_games)
        game.play(render=True, block_on_gameover=is_last)
        end_time = time.time()

        print(f"Game {i} finished in {end_time - start_time:.2f}s\n")

        # For non-last games we close the window immediately; for last,
        # play() already blocks until user closes, so this is harmless.
        if game.engine:
            try: game.engine.quit()
            except Exception: pass

        pygame.quit()
        if not is_last:
            time.sleep(1)




def main():
    # Choose players per side: "human", "minimax", or "stockfish"
    # Example: Minimax (white) vs Human (black)
    game = ChessGame(white_player="minimax", black_player="stockfish", minimax_depth=5)
    run_batch(num_games=10)

if __name__ == "__main__":
    main()

