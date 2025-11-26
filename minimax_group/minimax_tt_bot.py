import chess
import math
from enum import Enum

try:
    from .evaluate import evaluate
except ImportError:
    # Fallback for when running as a script
    try:
        from evaluate import evaluate
    except ImportError:
        evaluate = None


class TTEntryType(Enum):
    """Type of transposition table entry"""
    EXACT = 1      # Exact score (PV node)
    LOWER_BOUND = 2  # Alpha cutoff (fail-high, beta cutoff)
    UPPER_BOUND = 3  # Beta cutoff (fail-low, alpha unchanged)


class TTEntry:
    """
    Transposition Table Entry storing information about a previously evaluated position.
    
    Attributes:
        zobrist_key (int): Hash key of the position
        depth (int): Depth at which this position was evaluated
        score (float): Evaluation score
        flag (TTEntryType): Type of score (exact, lower bound, upper bound)
        best_move (chess.Move): Best move found at this position
    """
    def __init__(self, zobrist_key, depth, score, flag, best_move=None):
        self.zobrist_key = zobrist_key
        self.depth = depth
        self.score = score
        self.flag = flag
        self.best_move = best_move


class MinimaxBotWithTT:
    """
    A chess bot that uses Minimax with Alpha-Beta pruning and a Transposition Table.
    
    The Transposition Table (TT) is a hash table that stores previously evaluated positions.
    In chess, the same position can be reached through different move orders (transpositions).
    By caching these positions, we avoid re-evaluating them, which significantly speeds up search.
    
    Benefits of Transposition Table:
    1. Avoid redundant calculations for transposed positions
    2. Store best moves for move ordering (even more alpha-beta cutoffs)
    3. Can extend effective search depth
    4. Typical speedup: 2-10x depending on position
    
    Attributes:
        depth (int): Maximum search depth
        eval_fn: Evaluation function
        tt (dict): Transposition table (zobrist_hash -> TTEntry)
        tt_hits (int): Number of TT hits (for statistics)
        tt_stores (int): Number of TT stores
        nodes_searched (int): Number of nodes evaluated
    """
    
    def __init__(self, depth=3, eval_fn=None, tt_size_mb=128, use_null_move_pruning=True):
        """
        Initialize the Minimax bot with Transposition Table.
        
        Args:
            depth (int): Maximum search depth
            eval_fn: Evaluation function (defaults to comprehensive evaluation)
            tt_size_mb (int): Approximate size of TT in megabytes (not strictly enforced)
            use_null_move_pruning (bool): Whether to use null move pruning (default: True)
        """
        self.depth = depth
        self.eval_fn = eval_fn if eval_fn else self.default_evaluation
        self.tt = {}  # zobrist_hash -> TTEntry
        self.tt_size_mb = tt_size_mb
        # Rough estimate: each entry ~100 bytes, so max entries = (tt_size_mb * 1024 * 1024) / 100
        self.max_tt_entries = (tt_size_mb * 1024 * 1024) // 100
        self.use_null_move_pruning = use_null_move_pruning
        
        # Statistics
        self.tt_hits = 0
        self.tt_stores = 0
        self.nodes_searched = 0
        
    def clear_tt(self):
        """Clear the transposition table."""
        self.tt = {}
        self.tt_hits = 0
        self.tt_stores = 0
        
    def play(self, board):
        """
        Select the best move using minimax with alpha-beta pruning and transposition table.
        
        Args:
            board (chess.Board): Current chess board state
            
        Returns:
            chess.Move: Best move found
        """
        if board.is_game_over():
            return None
            
        # Reset statistics for this search
        self.nodes_searched = 0
        tt_hits_before = self.tt_hits
        
        best_move = None
        best_value = -math.inf
        alpha = -math.inf
        beta = math.inf
        
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None
        
        # Try to get move ordering from TT
        zobrist = chess.polyglot.zobrist_hash(board)
        if zobrist in self.tt:
            tt_entry = self.tt[zobrist]
            if tt_entry.best_move and tt_entry.best_move in legal_moves:
                # Try TT move first (likely to cause cutoff)
                legal_moves.remove(tt_entry.best_move)
                legal_moves.insert(0, tt_entry.best_move)
        
        # Search all moves
        for move in legal_moves:
            board.push(move)
            move_value = self.minimax(board, self.depth - 1, alpha, beta, False, allow_null=True)
            board.pop()
            
            if move_value > best_value:
                best_value = move_value
                best_move = move
            
            alpha = max(alpha, best_value)
            if beta <= alpha:
                break
        
        # Store root position in TT
        self.store_tt(zobrist, self.depth, best_value, TTEntryType.EXACT, best_move)
        
        tt_hits_this_search = self.tt_hits - tt_hits_before
        
        return best_move
    
    def minimax(self, board, depth, alpha, beta, maximizing_player, allow_null=True):
        """
        Minimax with alpha-beta pruning, transposition table, and null move pruning.
        
        Args:
            board (chess.Board): Current board state
            depth (int): Remaining depth
            alpha (float): Best value for maximizing player
            beta (float): Best value for minimizing player
            maximizing_player (bool): True if maximizing, False if minimizing
            allow_null (bool): Whether null move is allowed (prevents consecutive nulls)
            
        Returns:
            float: Evaluation score
        """
        self.nodes_searched += 1
        
        # Get zobrist hash for this position
        zobrist = chess.polyglot.zobrist_hash(board)
        original_alpha = alpha
        
        # Transposition table lookup
        if zobrist in self.tt:
            tt_entry = self.tt[zobrist]
            # Only use TT entry if it was searched to at least the same depth
            if tt_entry.depth >= depth:
                self.tt_hits += 1
                # Check the type of score and whether we can use it
                if tt_entry.flag == TTEntryType.EXACT:
                    return tt_entry.score
                elif tt_entry.flag == TTEntryType.LOWER_BOUND:
                    alpha = max(alpha, tt_entry.score)
                elif tt_entry.flag == TTEntryType.UPPER_BOUND:
                    beta = min(beta, tt_entry.score)
                
                # Check for cutoff
                if alpha >= beta:
                    return tt_entry.score
        
        # Terminal conditions
        if depth == 0 or board.is_game_over():
            score = self.eval_fn(board)
            self.store_tt(zobrist, depth, score, TTEntryType.EXACT)
            return score
        
        # Null move pruning
        # Only try null move if:
        # 1. Null move pruning is enabled
        # 2. We're not in check (can't pass when in check)
        # 3. Depth is sufficient (>= 3)
        # 4. allow_null is True (prevent consecutive null moves)
        if (self.use_null_move_pruning and allow_null and 
            not board.is_check() and depth >= 3):
            # Make a null move by switching turns
            board.push(chess.Move.null())
            # Search at reduced depth with reversed player
            null_score = self.minimax(board, depth - 3, -beta, -beta + 1, not maximizing_player, allow_null=False)
            board.pop()
            
            # If null move causes beta cutoff, prune this branch
            if maximizing_player:
                if null_score >= beta:
                    return beta
            else:
                if null_score <= alpha:
                    return alpha
        
        legal_moves = list(board.legal_moves)
        
        # Move ordering: try TT move first if available
        if zobrist in self.tt and self.tt[zobrist].best_move:
            tt_move = self.tt[zobrist].best_move
            if tt_move in legal_moves:
                legal_moves.remove(tt_move)
                legal_moves.insert(0, tt_move)
        
        best_move = None
        
        if maximizing_player:
            max_eval = -math.inf
            for move in legal_moves:
                board.push(move)
                eval_score = self.minimax(board, depth - 1, alpha, beta, False, allow_null=True)
                board.pop()
                
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = move
                
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break  # Beta cutoff
            
            # Store in TT
            if max_eval <= original_alpha:
                flag = TTEntryType.UPPER_BOUND  # Fail-low
            elif max_eval >= beta:
                flag = TTEntryType.LOWER_BOUND  # Fail-high
            else:
                flag = TTEntryType.EXACT  # PV node
            
            self.store_tt(zobrist, depth, max_eval, flag, best_move)
            return max_eval
        else:
            min_eval = math.inf
            for move in legal_moves:
                board.push(move)
                eval_score = self.minimax(board, depth - 1, alpha, beta, True, allow_null=True)
                board.pop()
                
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = move
                
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break  # Alpha cutoff
            
            # Store in TT
            if min_eval <= alpha:
                flag = TTEntryType.LOWER_BOUND  # Fail-high (from opponent's view)
            elif min_eval >= beta:
                flag = TTEntryType.UPPER_BOUND  # Fail-low (from opponent's view)
            else:
                flag = TTEntryType.EXACT  # PV node
            
            self.store_tt(zobrist, depth, min_eval, flag, best_move)
            return min_eval
    
    def store_tt(self, zobrist, depth, score, flag, best_move=None):
        """
        Store a position in the transposition table.
        
        Uses a replacement strategy: replace if:
        1. Position not in table yet
        2. New entry has greater depth (more reliable)
        3. Table is not at capacity
        
        Args:
            zobrist (int): Zobrist hash of position
            depth (int): Depth of search
            score (float): Evaluation score
            flag (TTEntryType): Type of score
            best_move (chess.Move): Best move found
        """
        # Check if we should store this entry
        if zobrist in self.tt:
            existing = self.tt[zobrist]
            # Replace if new depth is greater or equal (prefer recent searches)
            if depth >= existing.depth:
                self.tt[zobrist] = TTEntry(zobrist, depth, score, flag, best_move)
                self.tt_stores += 1
        else:
            # New entry - but check capacity
            if len(self.tt) < self.max_tt_entries:
                self.tt[zobrist] = TTEntry(zobrist, depth, score, flag, best_move)
                self.tt_stores += 1
            else:
                # Table full - could implement more sophisticated replacement
                # For now, just replace a random entry
                if len(self.tt) > 0:
                    # Simple strategy: replace the first entry (could be improved)
                    self.tt[zobrist] = TTEntry(zobrist, depth, score, flag, best_move)
                    self.tt_stores += 1
    
    def default_evaluation(self, board):
        """
        Default evaluation function (same as MinimaxBot).
        Returns score from perspective of side to move.
        """
        if board.is_checkmate():
            return -10000
        if board.is_stalemate():
            return 0
        if board.is_insufficient_material():
            return 0
        
        if evaluate is not None:
            white_score = evaluate(board)
            if board.turn == chess.WHITE:
                return white_score
            else:
                return -white_score
        else:
            # Fallback: simple material evaluation
            piece_values = {
                chess.PAWN: 100,
                chess.KNIGHT: 320,
                chess.BISHOP: 330,
                chess.ROOK: 500,
                chess.QUEEN: 900,
                chess.KING: 0
            }
            
            score = 0
            for square in chess.SQUARES:
                piece = board.piece_at(square)
                if piece:
                    value = piece_values[piece.piece_type]
                    if piece.color == chess.WHITE:
                        score += value
                    else:
                        score -= value
            
            if board.turn == chess.WHITE:
                return score
            else:
                return -score
    
    def get_stats(self):
        """Get statistics about TT performance."""
        return {
            'nodes_searched': self.nodes_searched,
            'tt_size': len(self.tt),
            'tt_hits': self.tt_hits,
            'tt_stores': self.tt_stores,
            'tt_hit_rate': (self.tt_hits / self.nodes_searched * 100) if self.nodes_searched > 0 else 0
        }



