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
import pandas as pd
import numpy as np


os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

df = pd.read_csv('positions_with_features_NEW.csv')


df = df[df['target'].abs() < 90]
df = df[df['target'].abs() > 1.0]

features = [
    # "attack_balance", #Y
    "attack_black",
    "attack_white",
    # "bishop_outposts_black", # Z
    # "bishop_outposts_white",
    "bishop_pair_white", # early
    "bishop_pair_black",
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
    "doubled_pawns_black",
    "doubled_pawns_white",
    "half_open_king_files", # late
    # "half_open_king_files_black",
    # "half_open_king_files_white",
    # "isolated_pawns", # mide - late
    "isolated_pawns_black",
    "isolated_pawns_white",
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
    "outposts_white", # zero
    "outposts_black", # zero
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

x = df[features]
y_raw = df['target']


scaler_x = StandardScaler()
# scaler_y = StandardScaler()

x_scaled = scaler_x.fit_transform(x)
# y_scaled = scaler_y.fit_transform(y_raw.values.reshape(-1, 1)).ravel()

model = Sequential([
    Input(shape=(len(features),)),
    Dense(256, activation='relu'),
    Dropout(0.15),
    Dense(128, activation='relu'),
    Dropout(0.15),
    Dense(64, activation='relu'),
    # Dropout(0.1),
    # Dense(36, activation='relu'),
    Dense(1)
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

print("Training model")
history = model.fit(
    x_scaled, y_raw,
    validation_split=0.2,
    epochs=50,
    batch_size=256,
    callbacks=[early_stop],
    verbose=1
)

x_train, x_test, y_train, y_test = train_test_split(x_scaled, y_raw, test_size=0.2, random_state=42)

y_pred_scaled = model.predict(x_test, verbose=0)
# y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
# y_test_orig = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()

r2 = r2_score(y_test, y_pred_scaled)
mse = mean_squared_error(y_test, y_pred_scaled)
rmse = np.sqrt(mse)


print("FINAL RESULT")
print(f"R²:  {r2:.4f} ")
print(f"RMSE: {rmse:.3f}")
print(f"MSE:  {mse:.3f}")
print(f"Stopped at epoch: {len(history.history['loss'])}")


save_dir = 'C:/Users/sasaa/OneDrive/Documents/GOLANG/src/MyVault/NOTES/UC-Davis/F25/ECS170/ChessAI/ann_group/'
os.makedirs(save_dir, exist_ok=True)

# model.save(save_dir + 'ann_model2.keras')
#
# joblib.dump(scaler_x, save_dir + 'scaler_features3.joblib')
# joblib.dump(scaler_y, save_dir + 'scaler_target3.joblib')
# joblib.dump(features, 'trained_feature_list3.joblib')



results_df = pd.DataFrame({
    'Metric': ['R² Score', 'MSE', 'RMSE'],
    'Value': [round(r2, 4), round(mse, 3), round(rmse, 3)]
})
# results_df.to_csv(save_dir + 'results_ann3.csv', index=False)


# BASELINE
# FINAL RESULT
# R²:  0.5291
# RMSE: 10.163
# MSE:  103.289
# Stopped at epoch: 28

# TARGET > 3
# R²:  0.5244
# RMSE: 16.462
# MSE:  271.005
# Stopped at epoch: 59

# TARGET >1 and TARGET < 90
# FINAL RESULT
# R²:  0.8971
# RMSE: 1.652
# MSE:  2.729
# Stopped at epoch: 29

# FINAL RESULT
# R²:  0.8838
# RMSE: 1.755
# MSE:  3.081
# Stopped at epoch: 27