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
from keras.src.callbacks import EarlyStopping
from scikeras.wrappers import KerasRegressor

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


df = pd.read_csv('C:/Users/sasaa/OneDrive/Documents/GOLANG/src/MyVault/NOTES/UC-Davis/F25/ECS170/ChessAI/ann_group/positions_with_features2.csv')

features = ["attack_balance","bishop_pair","bknrk_sqr_sum","center_attackers","connected_pawns",
            "defense_balance","doubled_pawns","half_open_king_files","king_ring_enemy_pressure",
            "material_bishop","material_knight","material_pawn","material_queen","material_rook",
            "mobility_balance","mobility_safe_balance","outposts","passed_pawns","pawn_shield",
            "pawn_sqr_sum","pieces_occupying_center","queen_sqr_sum","threat_balance"]

x = df[features]
y_raw = df['result_scaled'] # use feature 'target' instead

# normalize the response and predictors (separately)
scaler_x = StandardScaler()
scaler_y = StandardScaler()

x_scaled = scaler_x.fit_transform(x)
y_scaled = scaler_y.fit_transform(y_raw.values.reshape(-1, 1)).ravel()


def create_model(neurons1=512, neurons2=256, neurons3=128, activation='relu', dropout=0.3):
    model = Sequential([
        Input(shape=(len(features),)),
        Dense(neurons1, activation=activation),
        Dropout(dropout),
        Dense(neurons2, activation=activation),
        Dropout(dropout),
        Dense(neurons3, activation=activation),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

keras_model = KerasRegressor( model=create_model, neurons1=512, neurons2=256, neurons3=128, activation='relu', dropout=0.3, epochs=100, batch_size=256, verbose=0, callbacks=[EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)] )

param_grid = {
    'neurons1': [512, 768],
    'neurons2': [256, 384],
    'batch_size': [512],
    'activation': ['relu']
}

grid = GridSearchCV( estimator=keras_model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=1, verbose=1)

# Grid Search
print("GRID SEARCH")
grid_result = grid.fit(x_scaled, y_scaled)
cv_results = pd.DataFrame(grid_result.cv_results_)
cv_results['mean_test_mse'] = -cv_results['mean_test_score']
cv_results['mean_test_rmse'] = np.sqrt(cv_results['mean_test_mse'])

hyperparameters = ['activation', 'batch_size', 'neurons1', 'neurons2']
performance_data = []

for param in hyperparameters:
    param_key = f'param_{param}'
    if param_key not in cv_results.columns:
        continue
    grouped = cv_results.groupby(param_key).agg({
        'mean_test_mse': ['mean', 'std'],
        'mean_test_rmse': ['mean', 'std']
    }).reset_index()
    grouped.columns = ['value', 'avg_mse', 'std_mse', 'avg_rmse', 'std_rmse']
    grouped['hyperparameter'] = param
    performance_data.append(grouped[['hyperparameter', 'value', 'avg_mse', 'std_mse', 'avg_rmse', 'std_rmse']])

hyperparameter_performance = pd.concat(performance_data, ignore_index=True).round(3)
hyperparameter_performance.to_csv(
    'C:/Users/sasaa/OneDrive/Documents/GOLANG/src/MyVault/NOTES/UC-Davis/F25/ECS170/ChessAI/ann_group/hyperparams2_s.csv',
    index=False
)


# evaluate best model
best_model = grid_result.best_estimator_
best_params = grid_result.best_params_
best_mse_cv = -grid_result.best_score_
best_rmse_cv = np.sqrt(best_mse_cv)

# 80/20 test/train split
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y_scaled, test_size=0.2, random_state=42)
best_model.fit(x_train, y_train)
y_pred_scaled = best_model.predict(x_test)

# reverse scaling for response
y_pred_original = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
y_test_original = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()

# performance metrics
r2   = r2_score(y_test_original, y_pred_original)
mse  = mean_squared_error(y_test_original, y_pred_original)
rmse = np.sqrt(mse)


print("BEST HYPERPARAMETERS")
for k, v in best_params.items():
    print(f"{k}: {v}")

print("\nCross-validation (scaled space):")
print(f"Average MSE: {best_mse_cv:.3f}")
print(f"Average RMSE: {best_rmse_cv:.3f}")

print("\nFinal Test Set (original 0.5–35 scale):")
print(f"R² Score : {r2:.4f}")
print(f"RMSE     : {rmse:.3f} ")
print(f"MSE      : {mse:.3f}")
print()


# save performance metrics
results_df = pd.DataFrame({
    'Metric': ['Average MSE', 'Average RMSE', 'R² Score', 'MSE', 'RMSE'],
    'Value': [best_mse_cv, best_rmse_cv, r2, mse, rmse]
}).round(3)
results_df.to_csv('C:/Users/sasaa/OneDrive/Documents/GOLANG/src/MyVault/NOTES/UC-Davis/F25/ECS170/ChessAI/ann_group/results2_s.csv', index=False)

# save hyperparameters
params_df = pd.DataFrame({
    'Parameter': ['Neurons1', 'Neurons2', 'Activation', 'Learning Rate', 'Batch Size', 'Epochs'],
    'Value': [best_params.get('neurons1', 512), best_params.get('neurons2', 256),
              best_params.get('activation', 'relu'), '0.001 (Adam)', best_params.get('batch_size', 256), 'up to 1000']
})
params_df.to_csv('C:/Users/sasaa/OneDrive/Documents/GOLANG/src/MyVault/NOTES/UC-Davis/F25/ECS170/ChessAI/ann_group/params2_s.csv', index=False)

# save model and scalers
save_dir = 'C:/Users/sasaa/OneDrive/Documents/GOLANG/src/MyVault/NOTES/UC-Davis/F25/ECS170/ChessAI/ann_group/'
os.makedirs(save_dir, exist_ok=True)

joblib.dump(best_model, save_dir + 'ann_model2_s.joblib')
joblib.dump(scaler_x,   save_dir + 'scaler2_s.joblib')
joblib.dump(scaler_y,   save_dir + 'scaler2_target.joblib')


print("DONE")