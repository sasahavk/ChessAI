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

features = ["attack_balance","bishop_pair","bknrk_sqr_sum","center_attackers","connected_pawns",
            "defense_balance","doubled_pawns","half_open_king_files","king_ring_enemy_pressure",
            "material_bishop","material_knight","material_pawn","material_queen","material_rook",
            "mobility_balance","mobility_safe_balance","outposts","passed_pawns","pawn_shield",
            "pawn_sqr_sum","pieces_occupying_center","queen_sqr_sum","threat_balance"]

x = df[features]
y_raw = df['result_scaled']


scaler_x = StandardScaler()
scaler_y = StandardScaler()

x_scaled = scaler_x.fit_transform(x)
y_scaled = scaler_y.fit_transform(y_raw.values.reshape(-1, 1)).ravel()

model = Sequential([
    Input(shape=(len(features),)),
    Dense(768, activation='relu'),
    Dropout(0.3),
    Dense(384, activation='relu'),
    Dropout(0.3),
    Dense(192, activation='relu'),
    Dense(1)
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=50,
    restore_best_weights=True
)

print("Training model")
history = model.fit(
    x_scaled, y_scaled,
    validation_split=0.2,
    epochs=100,
    batch_size=512,
    callbacks=[early_stop],
    verbose=1
)

x_train, x_test, y_train, y_test = train_test_split(x_scaled, y_scaled, test_size=0.2, random_state=42)

y_pred_scaled = model.predict(x_test, verbose=0).ravel()
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
y_test_orig = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()

r2 = r2_score(y_test_orig, y_pred)
mse = mean_squared_error(y_test_orig, y_pred)
rmse = np.sqrt(mse)


print("FINAL RESULT")
print(f"R² Score  → {r2:.4f}    ← Should be 0.62–0.68")
print(f"RMSE      → {rmse:.3f}   ← Should be < 4.2")
print(f"MSE       → {mse:.3f}")
print(f"Stopped at epoch: {len(history.history['loss'])}")


# save_dir = 'C:/Users/sasaa/OneDrive/Documents/GOLANG/src/MyVault/NOTES/UC-Davis/F25/ECS170/ChessAI/ann_group/'
# os.makedirs(save_dir, exist_ok=True)
#
# model.save(save_dir + 'best_chess_ann_final.keras')
#
# # Save scalers
# joblib.dump(scaler_x, save_dir + 'scaler_features.joblib')
# joblib.dump(scaler_y, save_dir + 'scaler_target.joblib')
#
# # Save results (your format)
# results_df = pd.DataFrame({
#     'Metric': ['R² Score', 'MSE', 'RMSE'],
#     'Value': [round(r2, 4), round(mse, 3), round(rmse, 3)]
# })
# results_df.to_csv(save_dir + 'results_final.csv', index=False)
#
# print(f"\nModel saved → {save_dir}best_chess_ann_final.keras")
# print("You now have a REAL, strong chess evaluator!")