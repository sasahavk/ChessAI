import chess

debug:bool = False
def debugPrint(msg:str) -> None:
	if debug: print(msg)

MAX_VALUE:int = 999999999

pieceValues:dict = {
	chess.KING: MAX_VALUE,
	chess.QUEEN: 100,
	chess.KNIGHT: 70,
	chess.ROOK: 50,
	chess.BISHOP: 50,
	chess.PAWN: 10
}

pieceModValuables = {
	chess.KING: 1,
	chess.QUEEN: 3,
	chess.KNIGHT: 1,
	chess.ROOK: 1.5,
	chess.BISHOP: 1,
	chess.PAWN: 0.5
}

def evaluate(board:chess.Board) -> int:
	debugPrint("Board:")
	debugPrint(board)
	debugPrint("")

	debugPrint("White:" if board.turn == chess.WHITE else "Black:")
	myTotalValue:int = 0
	myMoves:list[chess.Move] = list(board.legal_moves)
	
	myTotalValue += getPotentialCaptureValue(board, myMoves)
	myTotalValue += getWholeBoardValue(board)
	

	board.turn = not board.turn
	debugPrint("\nWhite:" if board.turn == chess.WHITE else "\nBlack:")
	opponentTotalValue:int = 0
	opponentMoves:list[chess.Move]  = list(board.legal_moves)

	opponentTotalValue += getPotentialCaptureValue(board, opponentMoves, pieceModValuables)
	opponentTotalValue += getWholeBoardValue(board)

	return myTotalValue - opponentTotalValue

def getPotentialCaptureValue(board:chess.Board, moves:list[chess.Move], pieceModifiers:dict = None) -> int:
	total:int = 0
	
	for move in moves:
		if not board.is_capture(move):
			continue
		
		capturedPiece:chess.Piece = board.piece_at(move.to_square)
		capturedPieceValue:int = pieceValues[capturedPiece.piece_type]
		capturedPieceValue = capturedPieceValue * pieceModifiers[capturedPiece.piece_type] \
			if (pieceModifiers != None) else capturedPieceValue
		
		debugPrint(f"capturing: {capturedPiece.symbol()} ({capturedPieceValue}) | {move}")

		total += capturedPieceValue
	
	return total

def getWholeBoardValue(board:chess.Board):
	total:int = 0
	
	for square in chess.SQUARES:
		piece:chess.Piece = board.piece_at(square)
		if piece:
			total += pieceValues[piece.piece_type]
	
	return total

def runTestEvals() -> None:
	board:chess.Board = chess.Board()

	testBoards:list[str] = [
		{"fen": "k7/8/1r1n1q2/8/1p1Q1b2/8/8/7K", "turn": chess.BLACK},
		{"fen": "brkrqnnb/pppppppp/8/8/8/8/PPPPPPPP/BRKRQNNB", "turn": chess.WHITE},
		{"fen": "rnbqk2r/pppp1ppp/3b4/5n2/2N1p2P/4P3/PPPP1PP1/RNBQKB1R", "turn": chess.WHITE},
		{"fen": "rnbq1rk1/pp1p1ppp/2pn4/8/3Pp1QP/4P3/PPP2PP1/RNB1KB1R", "turn": chess.WHITE},
		{"fen": "rnb2rk1/pp1p1ppp/2pn4/3P4/2P1p1QP/4P3/P1qBKPP1/5B1R", "turn": chess.WHITE},
		{"fen": "rnb2rk1/pp1p1ppp/2pn4/3P3P/2P1p1Q1/4P3/P1qBKPP1/5B1R", "turn": chess.BLACK},
		{"fen": "rnb2rk1/pp1p1ppp/2p5/3P3P/2n1p1Q1/4P3/P1qBKPP1/5B1R", "turn": chess.WHITE},
		{"fen": "rnb2rk1/pp1p1ppp/2p4P/3P4/2n1p1Q1/4P3/P2qKPP1/5B1R", "turn": chess.BLACK}
	]

	i=1
	for config in testBoards:
		board.set_board_fen(config["fen"])
		board.turn = config["turn"]
		debugPrint(f"board {i} ({'White' if config["turn"] == chess.WHITE else 'Black'}): {evaluate(board)}\n===============")
		i += 1


if __name__ == "__main__":
	debug = True
	runTestEvals()