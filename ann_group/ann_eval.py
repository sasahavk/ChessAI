import chess
import pandas as pd
import joblib
import numpy as np
from keras.saving import load_model
import time
import warnings
warnings.filterwarnings("ignore",  message="X does not have valid feature names")
import feature_extractor
from feature_extractor import FeatureExtractorN
from minimax_group import minimax_bot

model_path = 'C:/Users/sasaa/OneDrive/Documents/GOLANG/src/MyVault/NOTES/UC-Davis/F25/ECS170/ChessAI/ann_group/ann_model2.keras'
scaler_x_path = 'C:/Users/sasaa/OneDrive/Documents/GOLANG/src/MyVault/NOTES/UC-Davis/F25/ECS170/ChessAI/ann_group/scaler_features2.joblib'
scaler_y_path = 'C:/Users/sasaa/OneDrive/Documents/GOLANG/src/MyVault/NOTES/UC-Davis/F25/ECS170/ChessAI/ann_group/scaler_target2.joblib'
trained_features_path ='C:/Users/sasaa/OneDrive/Documents/GOLANG/src/MyVault/NOTES/UC-Davis/F25/ECS170/ChessAI/ann_group/trained_feature_list2.joblib'


class ann:
    def __init__(self, model_path, scaler_x_path, scaler_y_path, trained_features_path):
        self.model = load_model(model_path)
        self.scaler_x = joblib.load(scaler_x_path)
        self.scaler_y = joblib.load(scaler_y_path)
        self.features = joblib.load(trained_features_path)
        self.fe = FeatureExtractorN(chess.Board(), feature_extractor.EARLY_GAME)
        self.features_dict = {}
        self.model_features_dict = {}
        self.df_input = pd.DataFrame()

    def eval(self, board: chess.Board) -> float:
        self.fe.set_board(board)
        self.features_dict = self.fe.get_features_subset_dict(self.features)
        X = np.array([self.features_dict.get(f, 0) for f in self.features],
                     dtype=np.float32).reshape(1, -1)

        X_scaled = self.scaler_x.transform(X)
        y_pred_scaled = self.model(X_scaled, training=False)
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled.numpy())[0]

        return float(y_pred.item())


sann = ann(model_path, scaler_x_path, scaler_y_path, trained_features_path)
# print(sann.eval(chess.Board("1r2kb1r/p1q2ppp/2p1p3/1bNpPn2/3P2P1/1P3N2/P4P1P/1RBQR1K1 b k - 0 16")))

for i in range(1,5):
    bot = minimax_bot.MinimaxBot(depth=i, eval_fn=sann.eval)
    start = time.perf_counter()
    move = bot.play(chess.Board("1r2kb1r/p1q2ppp/2p1p3/1bNpPn2/3P2P1/1P3N2/P4P1P/1RBQR1K1 b k - 0 16"))
    end = time.perf_counter()
    print(f"---Time ({i}: {end - start:.4f} seconds")
    print(f"---Move: {move}")


def benchmark_ann(ann_obj, num_positions=10_0):
    boards = []
    for _ in range(num_positions):
        board = chess.Board()
        for _ in range(np.random.randint(10, 61)):
            if board.is_game_over():
                break
            move = np.random.choice(list(board.legal_moves))
            board.push(move)
        boards.append(board.copy())

    _ = ann_obj.eval(boards[0])

    start_time = time.perf_counter()

    for board in boards:
        _ = ann_obj.eval(board)

    end_time = time.perf_counter()

    total_time = end_time - start_time
    avg_time_ms = (total_time / num_positions) * 1000
    predictions_per_second = num_positions / total_time

    print("=" * 60)
    print("PREDICTION SPEED BENCHMARK")
    print("=" * 60)
    print(f"Total time for {num_positions:,} positions: {total_time:.3f} seconds")
    print(f"Average time per position:          {avg_time_ms:.4f} ms")
    print(f"Predictions per second:            {predictions_per_second:.1f}")
    print(f"Model:                             {type(ann_obj).__name__}")
    print(f"Features:                          {len(ann_obj.features)}")
    print("=" * 60)

    if avg_time_ms < 1.0:
        print("LIGHTNING FAST — suitable for full engine search!")
    elif avg_time_ms < 5.0:
        print("Very fast — perfect for real-time play")
    elif avg_time_ms < 20.0:
        print("Good — usable with shallow search")
    else:
        print("Slow — needs optimization")

    return avg_time_ms

# sann = ann(model_path, scaler_x_path, scaler_y_path, trained_features_path)
# print(benchmark_ann(sann))

# Total time for 100 positions: 9.757 seconds
# Average time per position:          97.5678 ms
# Predictions per second:            10.2
# Model:                             ann
# Features:                          37

# Total time for 100 positions: 7.915 seconds
# Average time per position:          79.1539 ms
# Predictions per second:            12.6
# Model:                             ann
# Features:                          37

# Total time for 100 positions: 7.582 seconds
# Average time per position:          75.8235 ms
# Predictions per second:            13.2
# Model:                             ann
# Features:                          37

# Total time for 100 positions: 6.412 seconds
# Average time per position:          64.1232 ms
# Predictions per second:            15.6
# Model:                             ann
# Features:                          37

# Total time for 100 positions: 1.118 seconds
# Average time per position:          11.1816 ms
# Predictions per second:            89.4
# Model:                             ann
# Features:                          37