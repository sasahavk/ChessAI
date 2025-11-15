import chess
import math
try:
    from .evaluate import evaluate
except ImportError:
    # Fallback for when running as a script
    try:
        from evaluate import evaluate
    except ImportError:
        evaluate = None


class NegamaxBot:
    """
    A chess bot that uses the Negamax algorithm with Alpha-Beta pruning.
    
    Negamax is a variant of minimax that simplifies the algorithm by taking advantage 
    of the zero-sum property of chess: max(a, b) == -min(-a, -b).
    Instead of alternating between maximizing and minimizing, negamax always maximizes
    from the current player's perspective and negates the result when switching sides.
    
    Attributes:
        depth (int): The maximum depth to search in the game tree
        eval_fn: The evaluation function to score board positions
        nodes_searched (int): Counter for the number of nodes evaluated (for analysis)
    """
    
    def __init__(self, depth=3, eval_fn=None):
        """
        Initialize the Negamax bot.
        
        Args:
            depth (int): Maximum search depth (default: 3)
            eval_fn: Evaluation function that takes a board and returns a score.
                    If None, uses default evaluation function.
        """
        self.depth = depth
        self.eval_fn = eval_fn if eval_fn else self.default_evaluation
        self.nodes_searched = 0
        
    def play(self, board):
        """
        Select the best move using negamax with alpha-beta pruning.
        
        Args:
            board (chess.Board): The current chess board state
            
        Returns:
            chess.Move: The best move found by the algorithm
        """
        if board.is_game_over():
            return None
            
        self.nodes_searched = 0
        best_move = None
        best_value = -math.inf
        alpha = -math.inf
        beta = math.inf
        
        # Get all legal moves
        legal_moves = list(board.legal_moves)
        
        # If no legal moves, return None
        if not legal_moves:
            return None
        
        # Try each move and find the best one
        for move in legal_moves:
            board.push(move)
            # Negamax: negate the result from the opponent's perspective
            move_value = -self.negamax(board, self.depth - 1, -beta, -alpha)
            board.pop()
            
            # Update best move if this is better
            if move_value > best_value:
                best_value = move_value
                best_move = move
            
            # Alpha-beta pruning: update alpha
            alpha = max(alpha, move_value)
            if alpha >= beta:
                break  # Beta cutoff
        
        return best_move
    
    def negamax(self, board, depth, alpha, beta):
        """
        Negamax algorithm with alpha-beta pruning.
        
        The key insight of negamax is that in a zero-sum game:
        max(a, b) = -min(-a, -b)
        
        This means we can always maximize from the current player's perspective
        and simply negate the result when switching sides.
        
        Args:
            board (chess.Board): Current board state
            depth (int): Remaining search depth
            alpha (float): Best score we can guarantee (lower bound)
            beta (float): Opponent's best score they can guarantee (upper bound)
            
        Returns:
            float: The evaluated score of the position from the current player's perspective
        """
        self.nodes_searched += 1
        
        # Terminal conditions
        if depth == 0 or board.is_game_over():
            return self.eval_fn(board)
        
        legal_moves = list(board.legal_moves)
        max_value = -math.inf
        
        # Try each move
        for move in legal_moves:
            board.push(move)
            # Recursively evaluate position and negate (switch perspective)
            # Also negate and swap alpha/beta for the opponent
            value = -self.negamax(board, depth - 1, -beta, -alpha)
            board.pop()
            
            # Update maximum value found
            max_value = max(max_value, value)
            
            # Alpha-beta pruning
            alpha = max(alpha, value)
            if alpha >= beta:
                break  # Beta cutoff - prune remaining moves
        
        return max_value
    
    def default_evaluation(self, board):
        """
        Default evaluation function that scores board positions.
        Returns score from the perspective of the side to move.
        Positive scores favor the side to move, negative scores favor the opponent.
        
        Uses the comprehensive evaluation function from evaluate.py if available, which considers:
        - Material value (piece values)
        - Piece-square tables (positional value)
        - Pawn structure
        - Bishop pairs
        - Knight outposts
        - Rook files
        - King safety
        - Mobility
        
        Falls back to simple material evaluation if evaluate.py is not available.
        
        Args:
            board (chess.Board): The board to evaluate
            
        Returns:
            float: Evaluation score from the perspective of the side to move
        """
        # Terminal conditions
        if board.is_checkmate():
            # Checkmate: very bad for the side to move
            return -10000
        if board.is_stalemate():
            return 0
        if board.is_insufficient_material():
            return 0
        
        # Use the comprehensive evaluation function from evaluate.py if available
        if evaluate is not None:
            # It returns score from white's perspective
            white_score = evaluate(board)
            
            # Return score from the perspective of the side to move
            # If it's white's turn, return white_score (positive = good for white)
            # If it's black's turn, return -white_score (positive = good for black)
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
            
            # Return score from the perspective of the side to move
            if board.turn == chess.WHITE:
                return score
            else:
                return -score


