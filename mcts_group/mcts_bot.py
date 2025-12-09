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
		self.score:float = 0
		self.visits:int = 0
		self.lastMove:chess.Move = lastMove
		self.untried_moves:list[chess.Move] = list(board.legal_moves)

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
	def __init__(self, numRootSimulations:int, maxSimDepth:int, evalFunc=None, rememberPastBoardScores:bool=False, conditionForSimulatingTriedMoves=None, goForWin=False):
		self.numRootSimulations	:int = numRootSimulations	
		self.maxSimDepth:int = maxSimDepth
		self.evalFunc = backupEvalFunc if (evalFunc == None) else evalFunc
		self.color:bool = None
		self.rememberPastBoardScores:bool = rememberPastBoardScores
		self.boardScores:dict = {}
		self.goForWin:bool = goForWin
		self.conditionForSimulatingTriedMoves = defaultRuleForLookingAtTriedMoves \
			if (conditionForSimulatingTriedMoves == None) else conditionForSimulatingTriedMoves

	def play(self, board:chess.Board) -> chess.Move:		
		self.color = board.turn
		root:Node = Node(board)

		statSimDepthsReached:list[int] = []
		statNumPossibleWinningMoveSet:int = 0

		i:int = 0
		for i in range(self.numRootSimulations):
			# Selection + Expansion
			leaf:Node = self.applyTreePolicy(root, self.conditionForSimulatingTriedMoves(i))

			# Simulation
			result = None
			depth:int = None
			if self.rememberPastBoardScores and board.fen() in self.boardScores:
				result = self.boardScores[board.fen()]
				depth = 0
			else:
				result, depth = self.rollout(leaf)
			statSimDepthsReached.append(depth)

			# Backpropagation
			self.backpropagate(leaf, result)

			if self.goForWin and result >= VAL_WIN: 
				# print("YOU HAVE A WINNING SET OF MOVES!  TAKE IT!")
				break
			else:
				statNumPossibleWinningMoveSet += 1

		if not root.children:
			return random.choice(list(board.legal_moves))
		
		averageSimDepth:float = sum(statSimDepthsReached) / len(statSimDepthsReached)
		stats = {
			"avgSimDepth": averageSimDepth,
			"numRootSims": i,
			"boardEvalScore": None,
			"foundWinningMoveSets": statNumPossibleWinningMoveSet
		}
		
		bestChild = max(root.children, key=lambda n: n.score / n.visits)
		if bestChild.score < VAL_WIN and bestChild.score > VAL_LOSE:  # i.e didnt find an end state
			stats["boardEvalScore"] = bestChild.score
				
		return bestChild.lastMove, stats
	
	def applyTreePolicy(self, node:Node, skipUntriedMoves:bool = False) -> Node:
		currentNode:Node = node

		while not (currentNode.board.is_game_over()):
			if (currentNode.untried_moves and not skipUntriedMoves) or not currentNode.children: # "play" a random move
				randomMove:chess.Move = random.choice(currentNode.untried_moves)
				return currentNode.add_child(randomMove)
			currentNode:Node = currentNode.best_child()
		return currentNode
	
	def rollout(self, node:Node) -> (int, int):
		simBoard:chess.Board = node.board.copy()

		# "play" random moves until game over or simulation depth reached
		# make sure that if the loop breaks, it is our turn
		d:int = 0
		while d < self.maxSimDepth or simBoard.turn != self.color:
		# for d in range(self.maxSimDepth):
			if simBoard.is_game_over():
				result = simBoard.result()
				if result == "1-0":
					return (VAL_WIN if (self.color == chess.WHITE) else VAL_LOSE), d
				elif result == "0-1":
					return (VAL_LOSE if (self.color == chess.WHITE) else VAL_WIN), d
				else:
					return VAL_TIE, d
			
			currentLegalMoves:list[chess.Move] = list(simBoard.legal_moves)
			if not currentLegalMoves:
				break
			simBoard.push(random.choice(currentLegalMoves))
			d += 1

		# if max simulation depth reached, return board score based on evaluation
		boardScore:int = self.evalFunc(simBoard)  # score is from white's perspective
		return (boardScore if (self.color == chess.WHITE) else -boardScore), d
	
	def backpropagate(self, node:Node, score:int) -> None:
		currentNode:Node = node

		while currentNode != None:
			currentNode.visits += 1
			currentNode.score = score
			
			if self.rememberPastBoardScores and currentNode.parent != None:
				self.boardScores.update({currentNode.board.fen(): score})
			
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
    return score

def defaultRuleForLookingAtTriedMoves(totalRootSimulations:int) -> bool:
	# out of 10 root simulations
	# look at untried moves for 4 out of 10 moves
	# and tried moves for 10-4=6 out of 10 moves  
	numSimulationsNeededToSwitch:int = 4
	modulateTotalBy:int = 10
	return True if (totalRootSimulations % modulateTotalBy > numSimulationsNeededToSwitch) else False
