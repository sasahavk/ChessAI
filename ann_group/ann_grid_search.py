import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import SGD
from scikeras.wrappers import KerasRegressor
from keras.src.callbacks import EarlyStopping

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

df = pd.read_csv('C:/Users/sasaa/OneDrive/Documents/GOLANG/src/MyVault/NOTES/UC-Davis/F25/ECS170/ChessAI/ann_group/positions_with_features2.csv')
# select features and target
features =["attack_balance","bishop_pair","bknrk_sqr_sum","center_attackers","connected_pawns","defense_balance",
    "doubled_pawns","half_open_king_files","king_ring_enemy_pressure",
    "material_bishop","material_knight","material_pawn","material_queen","material_rook","mobility_balance","mobility_safe_balance", "outposts",
    "passed_pawns","pawn_shield","pawn_sqr_sum","pieces_occupying_center","queen_sqr_sum","threat_balance"
]

target = 'result_scaled' # use feature 'target' instead

# feature labels, will be used for coefficients table/file
feature_labels = {
    "attack_balance": "abal",
    "bishop_pair": "bbp",
    "bknrk_sqr_sum": "bknrkss",
    "center_attackers": "catk",
    "connected_pawns": "cpwn",
    "defense_balance": "dbal",
    "doubled_pawns": "dpwn",
    "half_open_king_files": "hogf",
    "king_ring_enemy_pressure": "krep",
    "material_bishop": "mb",
    "material_knight": "mk",
    "material_pawn": "mp",
    "material_queen": "mq",
    "material_rook": "mr",
    "mobility_balance": "mba",
    "mobility_safe_balance": "msba",
    "outposts":"outp",
    "passed_pawns": "ppwn",
    "pawn_shield": "pwns",
    "pawn_sqr_sum": "pss",
    "pieces_occupying_center": "pocc",
    "queen_sqr_sum": "qss",
    "threat_balance":"tbal",
}

x = df[features]
y = df[target]

# normalize features
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)


def create_model(neuron_count1=512, neuron_count2=256, neuron_count3 = 128, activation='relu', momentum=0.6):
    model = Sequential()
    model.add(Input(shape=(len(features),)))
    model.add(Dense(neuron_count1, activation=activation))
    model.add(Dense(neuron_count2, activation=activation))
    model.add(Dense(neuron_count3, activation=activation))
    model.add(Dense(1))
    optimizer = SGD(learning_rate=0.001, momentum=momentum)
    model.compile(optimizer=optimizer, loss='mse')
    return model

keras_model = KerasRegressor(model=create_model, activation='relu', verbose=0, momentum=0.6, neuron_count=256)
print(keras_model.get_params().keys())
param_grid = {
    'neuron_count1': [128, 256, 512],
    'neuron_count2': [128, 256, 512],
    'neuron_count3': [128, 256, 512],
    'activation': ['tanh', 'relu'],
    'momentum': [0.6, 0.9],
    'batch_size': [128, 256, 512],
}

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
grid = GridSearchCV(estimator=keras_model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=1,  verbose=0)
grid_result = grid.fit(x_scaled, y, callbacks=[early_stopping], validation_split=0.2)

# extras cv results for computing averages per hyperparameter value
cv_results = pd.DataFrame(grid_result.cv_results_)
cv_results['mean_test_mse'] = -cv_results['mean_test_score']
cv_results['mean_test_rmse'] = np.sqrt(cv_results['mean_test_mse'])
cv_results['std_test_mse'] = cv_results['std_test_score']
cv_results['std_test_rmse'] = cv_results['std_test_mse'] / (2 * cv_results['mean_test_rmse'])

hyperparameters = ['activation', 'momentum', 'neuron_count', 'batch_size']
performance_data = []

# group performance metrics for each hyperparameter (and their values)
for param in hyperparameters:
    param_key = f'param_{param}'
    grouped = cv_results.groupby(param_key).agg({
        'mean_test_mse': ['mean', 'std'],
        'mean_test_rmse': ['mean', 'std'],
    }).reset_index()
    grouped.columns = ['value', 'avg_mse', 'std_mse', 'avg_rmse', 'std_rmse']
    grouped['hyperparameter'] = param
    performance_data.append(grouped[['hyperparameter', 'value', 'avg_mse', 'std_mse', 'avg_rmse', 'std_rmse']])

hyperparameter_performance = pd.concat(performance_data, ignore_index=True)
hyperparameter_performance = hyperparameter_performance.round(3)

hyperparameter_performance = hyperparameter_performance[['hyperparameter', 'value', 'avg_mse', 'std_mse', 'avg_rmse', 'std_rmse']]

# print(hyperparameter_performance)

hyperparameter_performance.to_csv('C:/Users/sasaa/OneDrive/Documents/GOLANG/src/MyVault/NOTES/UC-Davis/F25/ECS170/ChessAI/ann_group/hyperparams_s.csv', index=False)
# # extract best model data
best_model = grid_result.best_estimator_
best_params = grid_result.best_params_
best_mse = -grid_result.best_score_
best_rmse = np.sqrt(best_mse)



# split train/test data into 80/20 split (final model)
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)

# train the model and make predictions for movie ratings
best_model.fit(x_train, y_train, callbacks=[early_stopping], validation_split=0.2)
y_pred = best_model.predict(x_test)

# calculate performance metrics
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("\nBest Hyperparameters:")
for param, value in best_params.items():
    print(f"{param}: {value}")
print("Epochs: 20 (fixed)")
print("Learning Rate: 0.01 (fixed)")

print("\nBest Model Performance:")
print(f"Average MSE: {best_mse:.3f}")
print(f"Average RMSE: {best_rmse:.3f}")

print("\nFinal Model Performance:")
print(f"R² Score: {r2:.3f}")
print(f"MSE: {mse:.3f}")
print(f"RMSE: {rmse:.3f}")

# create a dictionary to save average and final performance metrics
results = {
    'Metric': ['Average MSE', 'Average RMSE', ' R² Score', 'MSE', 'RMSE'],
    'Value': [best_mse, best_rmse, r2, mse, rmse]
}
# create a dictionary with best hyperparameter combination
params = {'Parameter': ['Neurons', 'Activation', 'Learning Rate', 'Momentum', 'Batch Size', 'Epochs'],
    'Value': [best_params['neuron_count'], best_params['activation'], 0.001, best_params['momentum'], best_params['batch_size'], 20]}

results_df = pd.DataFrame(results)
params_df = pd.DataFrame(params)
results_df['Value'] = results_df['Value'].round(3)

# save results and params as CSV file for later use
results_df.to_csv( 'C:/Users/sasaa/OneDrive/Documents/GOLANG/src/MyVault/NOTES/UC-Davis/F25/ECS170/ChessAI/ann_group/results_s.csv',index=False)
params_df.to_csv('C:/Users/sasaa/OneDrive/Documents/GOLANG/src/MyVault/NOTES/UC-Davis/F25/ECS170/ChessAI/ann_group/params_s.csv',index=False)

# save the models and its scaler
joblib.dump(best_model, 'C:/Users/sasaa/OneDrive/Documents/GOLANG/src/MyVault/NOTES/UC-Davis/F25/ECS170/ChessAI/ann_group/ann_model_s.joblib')
joblib.dump(scaler, 'C:/Users/sasaa/OneDrive/Documents/GOLANG/src/MyVault/NOTES/UC-Davis/F25/ECS170/ChessAI/ann_group/scaler_s.joblib')

