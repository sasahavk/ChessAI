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


class MinimaxBot:
    """
    A chess bot that uses the Minimax algorithm with Alpha-Beta pruning.
    
    Attributes:
        depth (int): The maximum depth to search in the game tree
        eval_fn: The evaluation function to score board positions
    """
    
    def __init__(self, depth=3, eval_fn=None):
        """
        Initialize the Minimax bot.
        
        Args:
            depth (int): Maximum search depth (default: 3)
            eval_fn: Evaluation function that takes a board and returns a score.
                    If None, uses default evaluation function.
        """
        self.depth = depth
        self.eval_fn = eval_fn if eval_fn else self.default_evaluation
        
    def play(self, board):
        """
        Select the best move using minimax with alpha-beta pruning.
        
        Args:
            board (chess.Board): The current chess board state
            
        Returns:
            chess.Move: The best move found by the algorithm
        """
        if board.is_game_over():
            return None
            
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
            # Minimize opponent's score (they will try to minimize our score)
            move_value = self.minimax(board, self.depth - 1, alpha, beta, False)
            board.pop()
            
            # Update best move if this is better
            if move_value > best_value:
                best_value = move_value
                best_move = move
            
            # Alpha-beta pruning: update alpha
            alpha = max(alpha, best_value)
            if beta <= alpha:
                break  # Beta cutoff
        
        return best_move
    
    def minimax(self, board, depth, alpha, beta, maximizing_player):
        """
        Minimax algorithm with alpha-beta pruning.
        
        Args:
            board (chess.Board): Current board state
            depth (int): Remaining search depth
            alpha (float): Best value for maximizing player (our best)
            beta (float): Best value for minimizing player (opponent's best)
            maximizing_player (bool): True if it's our turn, False if opponent's turn
            
        Returns:
            float: The evaluated score of the position
        """
        # Terminal conditions
        if depth == 0 or board.is_game_over():
            return self.eval_fn(board)
        
        legal_moves = list(board.legal_moves)
        
        if maximizing_player:
            # Our turn: maximize our score
            max_eval = -math.inf
            for move in legal_moves:
                board.push(move)
                eval_score = self.minimax(board, depth - 1, alpha, beta, False)
                board.pop()
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break  # Beta cutoff - prune remaining moves
            return max_eval
        else:
            # Opponent's turn: minimize our score (maximize their score)
            min_eval = math.inf
            for move in legal_moves:
                board.push(move)
                eval_score = self.minimax(board, depth - 1, alpha, beta, True)
                board.pop()
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break  # Alpha cutoff - prune remaining moves
            return min_eval
    
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


# Example usage and testing
if __name__ == "__main__":
    # Create a bot with depth 3
    bot = MinimaxBot(depth=3)
    
    # Test with a sample board
    board = chess.Board()
    print("Initial board:")
    print(board)
    print("\nBot's move:", bot.play(board))
    
    # Test with different depths
    for depth in [1, 2, 3]:
        bot = MinimaxBot(depth=depth)
        move = bot.play(board)
        print(f"Depth {depth}: {move}")

