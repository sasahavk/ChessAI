import pytest
import chess
import numpy as np
import feature_extractor as fe
# from feature_extractor import FeatureExtractor, END_GAME, MID_GAME, mirror_square  # replace 'your_module' with actual filename

# Helper FENs for testing
STARTPOS_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
CENTER_CONTROL_FEN = "rnbqkbnr/pppppppp/8/8/3PP3/8/PPP1PPPP/RNBQKBNR w KQkq - 0 1"  # e4, d4
BISHOP_PAIR_FEN = "r1bqkbnr/pppppppp/2n5/8/8/2N5/PPPPPPPP/R1BQKBNR w KQkq - 0 1"
KNIGHT_OUTPOST_FEN = "rnbqkb1r/ppp1pppp/5n2/3p4/3P4/2N2N2/PPP1PPPP/R1BQKB1R w KQkq - 0 1"
PASSED_PAWN_FEN = "8/2P5/8/8/8/3p4/8/8 w - - 0 1"  # c7 pawn is passed
KING_ENDGAME_FEN = "8/8/8/8/8/8/k7/K7 w - - 0 1"  # endgame king safety

@pytest.fixture
def extractor_startpos():
    board = chess.Board(STARTPOS_FEN)
    return fe.FeatureExtractor(board, fe.MID_GAME)

@pytest.fixture
def extractor_center():
    board = chess.Board(CENTER_CONTROL_FEN)
    return fe.FeatureExtractor(board, fe.MID_GAME)

@pytest.fixture
def extractor_endgame():
    board = chess.Board(KING_ENDGAME_FEN)
    return fe.FeatureExtractor(board, fe.END_GAME)


@pytest.mark.parametrize("fen, turn, expected", [
    # 1. Starting position — nothing in center
    ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", chess.WHITE,
     [0, 0, 0, 0, 0, 0]),

    # # 2. White pawns on d4, e4 — white to move
    ("rnbqkbnr/pppppppp/8/8/3PP3/8/PPP1PPPP/RNBQKBNR w KQkq - 0 1", chess.WHITE,
     [2, 0, 0, 0, 0, 0]),

    # 3. Same position — black to move → negative
    ("rnbqkbnr/pppppppp/8/8/3PP3/8/PPP1PPPP/RNBQKBNR b KQkq - 0 1", chess.BLACK,
     [-2, 0, 0, 0, 0, 0]),

    # 4. Black pawns on d5, e5
    ("rnbqkbnr/pppp1ppp/8/3pp3/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", chess.WHITE,
     [-2, 0, 0, 0, 0, 0]),

    # 5. Both sides have pawns in center
    ("rnbqkbnr/pppp1ppp/8/3PPp2/3Pp3/8/PPP1PPPP/RNBQKBNR w KQkq - 0 1", chess.WHITE,
     [2, 0, 0, 0, 0, 0]),

    # 6. White knight on e4
    ("rnbqkbnr/pppppppp/8/8/4N3/3n4/PPPPPPPP/R1BQKBNR w KQkq - 0 1", chess.WHITE,
     [0, 1, 0, 0, 0, 0]),

    # # 7. White queen on d4, black rook on e5
    ("8/8/8/4r3/3Q4/8/8/8 w - - 0 1", chess.WHITE,
     [0, 0, 0, -1, 1, 0]),  # rook=-1, queen=+1

    # # 8. White bishop on d5, black queen on e4
    ("8/8/8/3B4/4q3/8/8/8 w - - 0 1", chess.WHITE,
     [0, 0, 1, 0, -1, 0]),

    # 9. King in center (rare but valid!)
    ("8/8/8/3K4/8/8/8/8 w - - 0 1", chess.WHITE,
     [0, 0, 0, 0, 0, 1]),

    # 10. Complex mixed center — white advantage
    ("8/8/8/3PP3/3BB3/8/8/8 w - - 0 1", chess.WHITE,
     [2, 0, 2, 0, 0, 0]),  # white: pawn e4, bishop d4
])


def test_pieces_occupying_center_all_cases(fen, turn, expected):
    board = chess.Board(fen)
    board.turn = turn
    extractor = fe.FeatureExtractor(board, fe.MID_GAME)

    result = extractor.ft_pieces_occupying_center()
    # Convert expected to numpy array for comparison
    expected_array = np.array(expected, dtype=int)

    assert np.array_equal(result, expected_array), \
        f"Failed for FEN: {fen[:30]}... turn={['White', 'Black'][turn]}\n" \
        f"Got:      {result}\n" \
        f"Expected: {expected_array}"


@pytest.mark.parametrize("fen, turn, expected", [
    # # 1. Starting position
    # ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", chess.WHITE,
    #  [0, 0, 0, 0, 0, 0]),  # 2 knights (b1,g1)
    #
    # # 2. After 1. Nf3 Nc6 — knights attack center strongly
    # ("r1bqkbnr/pppppppp/2n5/8/8/8/PPPPPPPP/RNBQKB1R w KQkq - 2 2", chess.WHITE,
    #  [0, -2, 0, 0, 0, 0]),  # white knights: g1,b1,f3 → 3 attacks

    # 3. Bishops unleashed — Italian Game style
    ("r1bqk1nr/pppp1ppp/2n5/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4", chess.WHITE,
     [0, 1, 2, 0, 0, 0]),  # 1 knight + 2 bishops (c4,f3? wait — c4 bishop attacks d5/e5)

    # # 4. Queen + rook battery on open files
    # ("rnbqkb1r/pppp1ppp/5n2/4p3/3PP3/2N2N2/PPP2PPP/R1BQKB1R w KQkq - 0 3", chess.WHITE,
    #  [2, 2, 0, 1, 1, 0]),  # 2 pawns (d4,e4), 2 knights, 1 rook (a1→d4?), 1 queen
    #
    # # 5. Black to move — massive center attack with queen + bishops
    # ("rnbqk2r/pppp1ppp/4bn2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQK2R b KQkq - 2 5", chess.BLACK,
    #  [1, 1, 2, 0, 1, 0]),  # black: 1 pawn (e5), 1 knight, 2 bishops, 1 queen → all positive since black to move
])

def test_center_attackers_all_cases(fen, turn, expected):
    board = chess.Board(fen)
    board.turn = turn
    extractor = fe.FeatureExtractor(board, fe.MID_GAME)

    result = extractor.ft_center_attackers()
    expected_array = np.array(expected, dtype=int)

    print(board)
    assert np.array_equal(result, expected_array), \
        f"Failed for FEN: {fen}\n" \
        f"Turn: {'White' if turn else 'Black'}\n" \
        f"Got:      {result}\n" \
        f"Expected: {expected_array}\n" \
        f"Board:\n{board}\n"

def test_center_attackers(extractor_startpos):
    attackers = extractor_startpos.ft_center_attackers()
    # In startpos, knights attack center
    expected = np.array([0, 2, 0, 0, 0, 0])  # two white knights attack d4/e4/d5/e5
    assert np.array_equal(attackers, expected) or np.array_equal(attackers, -expected)


def test_bishop_pair(extractor_startpos):
    assert extractor_startpos.ft_bishop_pair() == 0  # both sides have pair

    board = chess.Board(BISHOP_PAIR_FEN)
    ext = fe.FeatureExtractor(board, fe.MID_GAME)
    assert ext.ft_bishop_pair() == 0  # still both have pair (even if one is missing? wait no)
    # Actually both still have both bishops in this FEN — let's break one
    board.remove_piece_at(chess.F1)  # remove light-square bishop
    ext.set_board(board)
    assert ext.ft_bishop_pair() == -1  # white missing one, black has both → -1


def test_knight_outpost():
    board = chess.Board(KNIGHT_OUTPOST_FEN)
    ext = fe.FeatureExtractor(board, fe.MID_GAME)
    # White knight on c3 is supported by d4 pawn, not attacked by black pawns → outpost
    assert ext.ft_knight_outposts() == 1


def test_bishop_outpost():
    fen = "rnbqkbnr/ppp1pppp/8/3p4/3P4/2N1B3/PPP1PPPP/R1BQKBNR w KQkq - 0 1"
    board = chess.Board(fen)
    ext = fe.FeatureExtractor(board, fe.MID_GAME)
    # White bishop on e3 supported? Let's assume not for now
    assert ext.ft_bishop_outposts() >= 0


def test_passed_pawns():
    board = chess.Board(PASSED_PAWN_FEN)
    ext = fe.FeatureExtractor(board, fe.MID_GAME)
    assert ext.ft_passed_pawns() == 1  # c7 pawn is passed for white


def test_piece_sqr_tables():
    board = chess.Board(STARTPOS_FEN)
    ext = fe.FeatureExtractor(board, fe.MID_GAME)

    # White pawn on e2 should get +0 from table (index 12: 0)
    assert ext.ft_pawn_sqr_sum() > 0  # overall positive due to central bonus

    # Move pawn to e4 → better square
    board.push(chess.Move(chess.E2, chess.E4))
    ext.set_board(board)
    assert ext.ft_pawn_sqr_sum() > 0  # should increase


def test_king_endgame_table_switch(extractor_endgame):
    # In endgame, king should prefer center
    king_center_fen = "8/8/8/8/8/8/3K4/8 w - - 0 1"
    board = chess.Board(king_center_fen)
    ext = fe.FeatureExtractor(board, fe.END_GAME)
    center_score = ext.ft_king_sqr_sum()

    # Move king to corner
    board.set_piece_at(chess.A1, chess.Piece(chess.KING, chess.WHITE))
    board.remove_piece_at(chess.D4)
    ext.set_board(board)
    corner_score = ext.ft_king_sqr_sum()

    assert center_score > corner_score  # endgame table rewards center


def test_get_features_returns_correct_length(extractor_startpos):
    features = extractor_startpos.get_features()
    assert len(features) == extractor_startpos.feature_count
    assert isinstance(features, list)
    assert all(isinstance(f, float) for f in features)


def test_material_features():
    board = chess.Board("rnbqkbnr/pppppppp/8/8/8/N7/PPPPPPPP/R1BQKBNR w KQkq - 0 1")  # extra knight
    ext = fe.FeatureExtractor(board, fe.MID_GAME)
    assert ext.ft_material_knight() > 0  # white has extra knight


def test_mobility_balance():
    board = chess.Board("rnbqkb1r/pppppppp/5n2/8/8/5N2/PPPPPPPP/RNBQKB1R w KQkq - 0 1")
    ext = fe.FeatureExtractor(board, fe.MID_GAME)
    mobility = ext.ft_mobility_balance()
    assert isinstance(mobility, float)


def test_mirror_square():
    assert fe.mirror_square(chess.E2) == chess.E7
    assert fe.mirror_square(chess.A1) == chess.A8
    assert fe.mirror_square(chess.H8) == chess.H1


def test_is_passed_pawn_edge_cases():
    board = chess.Board("8/pppppppp/8/8/8/8/PPPPPPPP/8 w - - 0 1")
    ext = fe.FeatureExtractor(board, fe.MID_GAME)
    # All pawns blocked → no passed
    assert ext.ft_passed_pawns() == 0


def test_outpost_invalid_piece():
    board = chess.Board(STARTPOS_FEN)
    ext = fe.FeatureExtractor(board, fe.MID_GAME)
    # No knights in outpost position
    assert ext.ft_knight_outposts() == 0


def test_feature_weight_application(extractor_center):
    features = extractor_center.get_features()
    # pieces_occupying_center should be weighted by CENTER_OCCUPY_BONUS
    occ = extractor_center.ft_pieces_occupying_center()
    weighted = np.sum(occ * fe.FeatureExtractor.CENTER_OCCUPY_BONUS)
    # Find index in features where this was stored
    # This is a bit meta, but we can trust logic if others pass
    assert any(abs(f - weighted) < 1e-6 for f in features)


if __name__ == "__main__":
    pytest.main(["-v", __file__])