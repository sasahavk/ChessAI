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

df = pd.read_csv('C:/Users/sasaa/OneDrive/Documents/GOLANG/src/MyVault/NOTES/UC-Davis/F25/ECS170/ChessAI/ann_group/positions_with_features2.csv')

# data filtering candidates
# df = df[df['move_num'] <= 40]
# df = df[df['target'].abs() > 1.0]

features = [
    "attack_balance",
    "bishop_pair_black",
    "bishop_pair_white",
    "bknrk_sqr_sum",
    "center_attackers_black",
    "center_attackers_white",
    "connected_pawns",
    "defense_balance",
    "doubled_pawns",
    "half_open_king_files",
    "king_ring_enemy_pressure",
    "material_bishop_black",
    "material_bishop_white",
    "material_knight_black",
    "material_knight_white",
    "material_pawn_black",
    "material_pawn_white",
    "material_queen_black",
    "material_queen_white",
    "material_rook_black",
    "material_rook_white",
    "mobility_balance",
    "mobility_safe_balance",
    "outposts_black",
    "outposts_white",
    "passed_pawns",
    "pawn_shield",
    "pawn_sqr_sum",
    "pieces_occupying_center",
    "queen_sqr_sum",
    "threat_balance"]

# 0.2167
# 0.3556

# count zeros per feature
zeros = (df[features] == 0).sum().sort_values(ascending=False)
percent_zeros = (zeros / len(df) * 100).round(1)

# count near-zeros  per features (|x| < 0.01 for floats, or just 0 for ints)
near_zeros = ((df[features].abs() < 0.01) | (df[features] == 0)).sum().sort_values(ascending=False)
percent_near_zeros = (near_zeros / len(df) * 100).round(1)

# print results
print("FEATURE ZERO ANALYSIS (new dataset)")
print("="*70)
print(f"Total positions: {len(df):,}")
print("\n% of positions where feature = exactly 0:")
print("-"*50)
for feat, pct in percent_zeros.items():
    print(f"{feat:30} {pct:5.1f}%  ({zeros[feat]:,} zeros)")

print("\n% of positions where feature ≈ 0 (dead/useless):")
print("-"*60)
for feat, pct in percent_near_zeros.items():
    star = " ← DEAD FEATURE" if pct > 85 else ""
    print(f"{feat:30} {pct:5.1f}%{star}")


x = df[features]
y_raw = df['target']
print(df['target'].value_counts())
print(df[features].describe())
scaler_x = StandardScaler()
scaler_y = StandardScaler()

x_scaled = scaler_x.fit_transform(x)
y_scaled = scaler_y.fit_transform(y_raw.values.reshape(-1, 1)).ravel()

# Simple linear model
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
