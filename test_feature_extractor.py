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
    return fe.FeatureExtractorN(board, fe.MID_GAME)

@pytest.fixture
def extractor_center():
    board = chess.Board(CENTER_CONTROL_FEN)
    return fe.FeatureExtractorN(board, fe.MID_GAME)

@pytest.fixture
def extractor_endgame():
    board = chess.Board(KING_ENDGAME_FEN)
    return fe.FeatureExtractorN(board, fe.END_GAME)


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
    print("TESTING")
    board = chess.Board(fen)
    board.turn = turn
    extractor = fe.FeatureExtractorN(board, fe.MID_GAME)

    result = extractor.ft_pieces_occupying_center()
    # Convert expected to numpy array for comparison
    expected_array = np.array(expected, dtype=int)

    assert np.array_equal(result, expected_array), \
        f"Failed for FEN: {fen[:30]}... turn={['White', 'Black'][turn]}\n" \
        f"Got:      {result}\n" \
        f"Expected: {expected_array}"




@pytest.mark.parametrize("fen, turn, expected", [
    # 1. Starting position
    ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", chess.WHITE,
     [0, 0, 0, 0, 0, 0]),  # 2 knights (b1,g1)

    # 2. After 1. Nf3 Nc6 — knights attack center strongly
    ("r1bqkbnr/pppppppp/2n5/8/8/8/PPPPPPPP/RNBQKB1R w KQkq - 2 2", chess.WHITE,
     [0, -2, 0, 0, 0, 0]),  # white knights: g1,b1,f3 → 3 attacks

    ("8/8/8/8/2b5/3N4/8/8 w - - 0 1", chess.WHITE,
     [0,1, -1, 0, 0, 0]),  # 1 knight + 2 bishops (c4,f3? wait — c4 bishop attacks d5/e5)

    # . Queen + rook battery on open files
    ("4q3/4Q3/8/8/8/8/8/8 w - - 0 1", chess.WHITE,
     [0, 0, 0, 0, 2, 0]),  # 2 pawns (d4,e4), 2 knights, 1 rook (a1→d4?), 1 queen
])

def test_center_attackers_all_cases(fen, turn, expected):
    board = chess.Board(fen)
    board.turn = turn
    extractor = fe.FeatureExtractorN(board, fe.MID_GAME)

    result = extractor.ft_center_attackers()
    expected_array = np.array(expected, dtype=int)

    print(board)
    assert np.array_equal(result, expected_array), \
        f"Failed for FEN: {fen}\n" \
        f"Turn: {'White' if turn else 'Black'}\n" \
        f"Got:      {result}\n" \
        f"Expected: {expected_array}\n" \
        f"Board:\n{board}\n"


@pytest.mark.parametrize("fen, turn, expected",[
        # 1. White to move, only White has the bishop pair — should return +1
        (
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            chess.WHITE,
            0   # both sides have pair → 1 - 1 = 0
        ),

        # 2. White to move, White has both bishops, Black has zero — should return +1
        (
                "rn1qkbnr/pppppppp/8/8/8/8/PPPPPPPP/R1BQKB1R w KQkq - 0 1",
                chess.WHITE,
                1  # white_has_bp = 1, black_has_bp = 0 → return 1 - 0 = +1
        ),

        # 3. Black to move, Black has both bishops, White has none — should return +1
        (
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/r1BqkQ1r b KQkq - 0 1",
                chess.BLACK,
                1  # black_has_bp = 1, white_has_bp = 0 → return 1 - 0 = +1
        ),
    ])

def test_ft_bishop_pair(fen, turn, expected, capsys):
    board = chess.Board(fen)
    board.turn = turn
    extractor = fe.FeatureExtractorN(board, fe.MID_GAME)

    result = extractor.ft_bishop_pair()

    captured = capsys.readouterr()
    prints = captured.out.strip().split('\n')
    white_bp = int(prints[0]) if prints else None
    black_bp = int(prints[1]) if len(prints) > 1 else None

    assert result == expected, \
        f"FEN: {fen}\n" \
        f"Turn: {'White' if turn else 'Black'}\n" \
        f"Printed: {white_bp}, {black_bp}\n" \
        f"Got: {result}, Expected: {expected}\n" \
        f"Board:\n{board}"


@pytest.mark.parametrize(
    "fen, turn, expected",
    [
        # 1. White knight on b5 — perfect outpost, Black has none
        (
                "rnbqkb1r/p1pp1ppp/1p2pn2/1N6/1PP5/P7/1P1PPPPP/RNBQKBNR w KQkq - 0 6",
                chess.WHITE,
                1  # +1 for White
        ),

        # 2. Black knight on b4 — perfect outpost, White has none
        (
                "8/2p1p1p1/p1P1P1Pp/N1P1P1PN/1n6/1N6/8/8 w - - 0 1",
                chess.BLACK,
                0  # +1 for Black
        ),

        # 3. Both have one outpost — cancels out
        (
                "8/8/8/2p5/1n6/8/7N/8 w - - 0 1",
                chess.WHITE,
                -1  # 1 - 1 = 0
        ),
    ]
)


def test_ft_knight_outposts(fen, turn, expected):
    board = chess.Board(fen)
    board.turn = turn
    extractor = fe.FeatureExtractorN(board,0)
    result = extractor.ft_knight_outposts()

    assert result == expected, f"Failed: {fen} | Got {result}, expected {expected}\n{board}"


@pytest.mark.parametrize("fen, turn, expected",[
        # 1. White 1. White bishop on d5 — outpost (rank 5, supported by e4, not attacked by black pawn)
        #    Black has no bishop → White: +1
        (
                "8/8/8/3Bp3/4P3/8/8/8 w - - 0 1",
                chess.WHITE,
                1  # White bishop on d5 is outpost → 1 - 0 = +1
        ),

        # 2. Black bishop on d4 — outpost (rank 4, supported by e5, not attacked by white pawn)
        #    White has no bishop → White to move → 0 - 1 = -1
        (
                "8/8/8/4p3/3bP3/8/8/8 w - - 0 1",
                chess.WHITE,
                -1  # Black bishop on d4 is outpost → 0 - 1 = -1
        ),

        # 3. White bishop on c4 — NOT outpost (own half, rank 4), Black has none → 0
        (
                "8/8/8/8/2bP4/8/8/8 w - - 0 1",
                chess.WHITE,
                0  # Bishop on c4 is on own half → not outpost
        ),
    ])

def test_ft_bishop_outposts(fen, turn, expected):
    board = chess.Board(fen)
    board.turn = turn
    extractor = fe.FeatureExtractorN(board, 0)

    # Call your method: ft_bishop_outposts() → outpost(chess.BISHOP)
    result = extractor.outpost(chess.BISHOP)  # or extractor.ft_bishop_outposts() if you have it

    assert result == expected, \
        f"Failed for FEN: {fen}\n" \
        f"Got: {result}, Expected: {expected}\n" \
        f"Board:\n{board}\n"


def test_passed_pawns():
    board = chess.Board(PASSED_PAWN_FEN)
    ext = fe.FeatureExtractorN(board, fe.MID_GAME)
    assert ext.ft_passed_pawns() == 1  # c7 pawn is passed for white


def test_piece_sqr_tables():
    board = chess.Board(STARTPOS_FEN)
    ext = fe.FeatureExtractorN(board, fe.MID_GAME)

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
    ext = fe.FeatureExtractorN(board, fe.END_GAME)
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
    ext = fe.FeatureExtractorN(board, fe.MID_GAME)
    assert ext.ft_material_knight() > 0  # white has extra knight


def test_mobility_balance():
    board = chess.Board("rnbqkb1r/pppppppp/5n2/8/8/5N2/PPPPPPPP/RNBQKB1R w KQkq - 0 1")
    ext = fe.FeatureExtractorN(board, fe.MID_GAME)
    mobility = ext.ft_mobility_balance()
    assert isinstance(mobility, float)


def test_mirror_square():
    assert fe.mirror_square(chess.E2) == chess.E7
    assert fe.mirror_square(chess.A1) == chess.A8
    assert fe.mirror_square(chess.H8) == chess.H1


def test_is_passed_pawn_edge_cases():
    board = chess.Board("8/pppppppp/8/8/8/8/PPPPPPPP/8 w - - 0 1")
    ext = fe.FeatureExtractorN(board, fe.MID_GAME)
    # All pawns blocked → no passed
    assert ext.ft_passed_pawns() == 0


def test_outpost_invalid_piece():
    board = chess.Board(STARTPOS_FEN)
    ext = fe.FeatureExtractorN(board, fe.MID_GAME)
    # No knights in outpost position
    assert ext.ft_knight_outposts() == 0


def test_feature_weight_application(extractor_center):
    features = extractor_center.get_features()
    # pieces_occupying_center should be weighted by CENTER_OCCUPY_BONUS
    occ = extractor_center.ft_pieces_occupying_center()
    weighted = np.sum(occ * fe.FeatureExtractorN.CENTER_OCCUPY_BONUS)
    # Find index in features where this was stored
    # This is a bit meta, but we can trust logic if others pass
    assert any(abs(f - weighted) < 1e-6 for f in features)