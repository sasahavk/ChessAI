import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.optimizers import Adam
from scikeras.wrappers import KerasRegressor
from keras.src.callbacks import EarlyStopping

df = pd.read_csv('positions_with_features_NEW.csv')
print(df.shape[0])

# filter checkmate positions and approx. even positions
df = df[df['target'].abs() < 90]
df = df[df['target'].abs() > 1.0]
print(df.shape[0])      # → number of rows

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

# normalize features
scaler_x = StandardScaler()
x_scaled = scaler_x.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x_scaled, y_raw, test_size=0.2, random_state=42)

def create_model(neuron_count=512, activation='relu', dropout=0.1, learning_rate=0.001):
    model = Sequential([
        Input(shape=(len(features),)),
        Dense(neuron_count, activation=activation),
        Dropout(dropout),
        Dense(neuron_count//2, activation=activation),
        Dropout(dropout),
        Dense(neuron_count//4, activation=activation),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
    return model

keras_model = KerasRegressor(
    model=create_model,
    epochs=20,
    batch_size=256,
    verbose=1,
    activation='relu',
    dropout=0.1,
    neuron_count=512,
    learning_rate=0.001
)
param_grid = {
    'neuron_count': [256, 512],
    'dropout':[0.1, 0.2],
    'learning_rate':[0.01, 0.001],
    'activation': ['tanh', 'relu'],
}

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
grid = GridSearchCV(estimator=keras_model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=1,  verbose=1)
grid_result = grid.fit(x_train, y_train, callbacks=[early_stopping], validation_split=0.2)

# extras cv results for computing averages per hyperparameter value
cv_results = pd.DataFrame(grid_result.cv_results_)
print(cv_results)
cv_results['MSE'] = -cv_results['mean_test_score']
cv_results['RMSE'] = np.sqrt(cv_results['MSE'])
cv_results['R2'] = 1 - (cv_results['MSE'] / np.var(y_raw))


results_all = cv_results[[
    'param_neuron_count',
    'param_dropout',
    'param_learning_rate',
    'param_activation',
    'MSE',
    'RMSE',
    'R2'
]].copy()

results_all.columns = ['neurons', 'dropout', 'learning_rate', 'activation' ,'MSE', 'RMSE', 'R2']
results_all = results_all.round(6)

best_model = grid_result.best_estimator_.model_
best_idx = grid_result.best_index_
best_r2 = results_all.loc[best_idx, 'R2']
best_rmse = results_all.loc[best_idx, 'RMSE']
best_mse = results_all.loc[best_idx, 'MSE']

save_dir = 'C:/Users/sasaa/OneDrive/Documents/GOLANG/src/MyVault/NOTES/UC-Davis/F25/ECS170/ChessAI/ann_group/'
results_all.to_csv(save_dir + 'hyperparameter_results3.csv', index=False)

# best_summary = pd.DataFrame({
#     'Metric': ['R2', 'RMSE', 'MSE'],
#     'Best_Value': [best_r2, best_rmse, best_mse],
#     'Best_Params': [str(grid_result.best_params_)] * 3
# })
# best_summary.to_csv(save_dir + 'best_model_metrics.csv', index=False)


print("="*70)
print("GRID SEARCH SUMMARY")
print("="*70)
print(results_all.sort_values('R2', ascending=False).head(10))
print("\n" + "="*70)
print("BEST MODEL")
print("="*70)
print(f"Best params: {grid_result.best_params_}")
print(f"Best R² : {best_r2:.6f}")
print(f"Best RMSE: {best_rmse:.6f}")
print(f"Best MSE : {best_mse:.6f}")
print("="*70)

# save the models and its scaler
best_model.save(save_dir + 'ann_model3.keras')
joblib.dump(scaler_x, save_dir + 'scaler_features3.joblib')
joblib.dump(features, 'trained_feature_list3.joblib')

# Use the BEST model from grid search
final_model = grid_result.best_estimator_.model_   # ← the actual trained Keras model

# Final prediction on test set
y_pred = final_model.predict(x_test, verbose=0).ravel()

# Metrics
final_r2 = r2_score(y_test, y_pred)
final_mse = mean_squared_error(y_test, y_pred)
final_rmse = np.sqrt(final_mse)


print(f"FINAL TEST SET PERFORMANCE:")
print(f"   R²   = {final_r2:.5f}")
print(f"   RMSE = {final_rmse:.4f}")
print(f"   MSE  = {final_mse:.6f}")

# Optional: save final test results
results_final = pd.DataFrame([{
    'R2': final_r2,
    'RMSE': final_rmse,
    'MSE': final_mse,
    'Best_Params': str(grid_result.best_params_)
}])
results_final.to_csv(save_dir + 'best_model_results3.csv', index=False)

# 224586
# 120574