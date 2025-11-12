import chess
import math, random

VAL_WIN:int = 9999999
VAL_LOSE:int = -9999999
VAL_TIE:int = 0

class Node:
	def __init__(self, board:chess.Board, parent:"Node"=None, lastMove:chess.Move=None):
		self.board:chess.Board = board
		self.parent:"Node" = parent
		self.children:list["Node"] = []
		# self.wins:float = 0.0
		self.score:int = 0
		self.visits:int = 0
		self.lastMove:chess.Move = lastMove
		self.untried_moves:list[chess.Move] = list(board.legal_moves)
		# self.player_turn = board.turn  # True=White, False=Black

	def ucb1(self) -> float:
		if self.visits == 0:
			return float("inf")
	
		sqrtOfTwo:float = 1.41421356
		return (self.score / self.visits) + sqrtOfTwo * math.sqrt(math.log(self.parent.visits) / self.visits)

	def best_child(self) -> "Node":
		return max(self.children, key=lambda n: n.ucb1())

	def add_child(self, move:chess.Move) -> "Node":
		newBoard:chess.Board = self.board.copy()

		newBoard.push(move)  # performs move on the board
		child:"Node" = Node(newBoard, parent=self, lastMove=move)

		self.untried_moves.remove(move)
		self.children.append(child)
		return child
	

class MonteCarloSearchTreeBot:
	def __init__(self, numRootSimulations:int, maxSimDepth:int, evalFunc=None):
		self.numRootSimulations	:int = numRootSimulations	
		self.maxSimDepth:int = maxSimDepth
		self.evalFunc = backupEvalFunc if (evalFunc == None) else evalFunc

	def play(self, board:chess.Board) -> chess.Move:
		root:Node = Node(board)

		for _ in range(self.numRootSimulations):
			# Selection + Expansion
			leaf:Node = self.applyTreePolicy(root)

			# Simulation
			result:int = self.rollout(leaf)

			# Backpropagation
			self.backpropagate(leaf, result)

		if not root.children:
			return random.choice(list(board.legal_moves))
		
		return max(root.children, key=lambda n: n.score).lastMove
	
	def applyTreePolicy(self, node:Node) -> Node:
		currentNode:Node = node

		while not (currentNode.board.is_game_over()):
			if currentNode.untried_moves: # "play" a random move
				randomMove:chess.Move = random.choice(node.untried_moves)
				return currentNode.add_child(randomMove)
			currentNode = currentNode.best_child()
		return currentNode
	
	def rollout(self, node:Node) -> int:
		simBoard:chess.Board = node.board.copy()

		# "play" random moves until game over or simulation depth reached
		for _ in range(self.maxSimDepth):
			if simBoard.is_game_over():
				result = simBoard.result()
				if result == "1-0": return VAL_WIN
				elif result == "0-1": return VAL_LOSE
				else: return VAL_TIE
			
			currentLegalMoves:list[chess.Move] = list(simBoard.legal_moves)
			if not currentLegalMoves:
				break
			simBoard.push(random.choice(currentLegalMoves))

		# if max simulation depth reached, return board score based on evaluation
		return self.evalFunc(simBoard)
	
	def backpropagate(self, node:Node, score:int) -> None:
		currentNode:Node = node

		while currentNode != None:
			currentNode.visits += 1
			currentNode.score = score
			currentNode = currentNode.parent

def backupEvalFunc(board:chess.Board) -> int:
    vals:dict = {
		chess.PAWN:1,
		chess.KNIGHT:3,
		chess.BISHOP:3,
		chess.ROOK:5,
		chess.QUEEN:9
	}
    score:int = 0
    for piece,value in vals.items():
        score += value * (
			len(board.pieces(piece, chess.WHITE))
			- len(board.pieces(piece, chess.BLACK))
		)
    return score if board.turn else -score



