from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.optimizers import Adam
from keras.src.callbacks import EarlyStopping

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

df = pd.read_csv('/ann_group/positions_with_features_NEW.csv')

# data filtering candidates
# df = df[df['move_num'] <= 40]
df = df[df['target'].abs() > 3.0]

features = [
    # "attack_balance", #Y
    "attack_black",
    "attack_white",
    # "bishop_outposts_black", # Z
    # "bishop_outposts_white",
    # "bishop_pair_white", # early
    # "bishop_pair_black",
    # "bishop_sqr_sum", #Y
    # "bishop_sqr_sum_black",
    # "bishop_sqr_sum_white",
    "center_attackers_white", #Y
    "center_attackers_black", #Y
    "connected_pawns", # Y
    # "connected_pawns_black",
    # "connected_pawns_white",
    "defense_balance", # Y
    # "defense_black",
    # "defense_white",
    # "doubled_pawns", # zero
    # "doubled_pawns_black",
    # "doubled_pawns_white",
    "half_open_king_files", # late
    # "half_open_king_files_black",
    # "half_open_king_files_white",
    # "isolated_pawns", # mide - late
    # "isolated_pawns_black",
    # "isolated_pawns_white",
    "king_ring_enemy_pressure", # Y
    # "king_ring_enemy_pressure_black",
    # "king_ring_enemy_pressure_white",
    # "king_sqr_sum",
    # "king_sqr_sum_black", # Y
    # "king_sqr_sum_white", # Y
    # "knight_outposts_black", # zero
    # "knight_outposts_white", # zero
    # "knight_sqr_sum", # early - mid
    # "knight_sqr_sum_black",
    # "knight_sqr_sum_white",
    "material_bishop_white", # early - mid
    "material_bishop_black", # early - mid
    "material_knight_white", # early - mid
    "material_knight_black", # early - mid
    "material_pawn_white", # Y
    "material_pawn_black", # Y
    "material_queen_white", # early - mid
    "material_queen_black", # early - mid
    "material_rook_white", # early - mid
    "material_rook_black", # early - mid
    # "mobility_balance", # Y
    "mobility_black",
    "mobility_white",
    # "mobility_safe_balance", # Y
    "mobility_safe_black",
    "mobility_safe_white",
    # "outposts_white", # zero
    # "outposts_black", # zero
    "passed_pawns", # mid - late
    # "passed_pawns_black",
    # "passed_pawns_white",
    "pawn_shield", # Y
    # "pawn_shield_black",
    # "pawn_shield_white",
    # "pawn_sqr_sum", # Y
    # "pawn_sqr_sum_black",
    # "pawn_sqr_sum_white",
    "pieces_occupying_center", # Y
    # "queen_sqr_sum", # zero
    # "queen_sqr_sum_black",
    # "queen_sqr_sum_white",
    # "rook_sqr_sum", # zero
    # "rook_sqr_sum_black",
    # "rook_sqr_sum_white",
    "sqr_sum_black",
    "sqr_sum_white",
    # "threat_balance",
    "threat_black", # Y
    "threat_white" # Y
]

# 0.2167
# 0.3556

x = df[features]
y_raw = df['target']

scaler_x = StandardScaler()
scaler_y = StandardScaler()

x_scaled = scaler_x.fit_transform(x)
y_scaled = scaler_y.fit_transform(y_raw.values.reshape(-1, 1)).ravel()

linear = Ridge(alpha=1.0)
linear.fit(x_scaled, y_scaled)

pred_lin = linear.predict(x_scaled)
r2_linear = r2_score(y_scaled, pred_lin)

pred_lin_orig = scaler_y.inverse_transform(pred_lin.reshape(-1,1)).ravel()
true_orig = scaler_y.inverse_transform(y_scaled.reshape(-1,1)).ravel()
rmse_linear = np.sqrt(mean_squared_error(true_orig, pred_lin_orig))

print(f"Linear model R² (scaled): {r2_linear:.4f}")
print(f"Linear model R² (original scale): {r2_score(true_orig, pred_lin_orig):.4f}")
print(f"Linear model RMSE (original): {rmse_linear:.3f}")

# BASELINE
# Linear model R² (scaled): 0.3240
# Linear model R² (original scale): 0.3240
# Linear model RMSE (original): 12.091

# TARGET > 1.0
# Linear model R² (scaled): 0.3395
# Linear model R² (original scale): 0.3395
# Linear model RMSE (original): 15.920

# TARGET > 2.0 (98,092)
# Linear model R² (scaled): 0.3444
# Linear model R² (original scale): 0.3444
# Linear model RMSE (original): 17.852

# TARGET > 3.0 (81,049)
# Linear model R² (scaled): 0.3479
# Linear model R² (original scale): 0.3479
# Linear model RMSE (original): 19.508

# TARGET > 4.0 (67,213)
# Linear model R² (scaled): 0.3517
# Linear model R² (original scale): 0.3517
# Linear model RMSE (original): 21.247

# TARGET > 5.0 (52,470)
# Linear model R² (scaled): 0.3581
# Linear model R² (original scale): 0.3581
# Linear model RMSE (original): 23.724

# TARGET > 6.0 (39,036)
# Linear model R² (scaled): 0.3702
# Linear model R² (original scale): 0.3702
# Linear model RMSE (original): 26.912