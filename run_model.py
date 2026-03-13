import sys
import os
import json
import numpy as np
import pandas as pd
import pandas_datareader.data as web
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score

from modelos.modelo_complejo import build_complex_model

print("1. Cargando datos...")
df_ret = pd.read_csv('datos/dataset_ia_log_returns_10y.csv', index_col=0, parse_dates=True)

df_vol = df_ret.rolling(window=21).std()

fecha_inicio = df_ret.index.min()
fecha_fin = df_ret.index.max()
df_fed = web.DataReader('DFF', 'fred', fecha_inicio, fecha_fin)
df_fed.rename(columns={'DFF': 'Tipo_Interes_FED'}, inplace=True)
df_fed['Tipo_Interes_FED'] = df_fed['Tipo_Interes_FED'] / 100 

df_master = df_ret.join(df_vol, rsuffix='_vol').join(df_fed, how='left')
df_master['Tipo_Interes_FED'] = df_master['Tipo_Interes_FED'].ffill() 
df_master = df_master.dropna() 

datos_secuenciales = df_master.drop(columns=['Tipo_Interes_FED']).values 
datos_macro = df_master[['Tipo_Interes_FED']].values                     
datos_y = df_master.iloc[:, :12].values                                  

print("2. Escalando datos...")
scaler_seq = MinMaxScaler(feature_range=(-1, 1))
scaler_y = MinMaxScaler(feature_range=(-1, 1))

scaled_seq = scaler_seq.fit_transform(datos_secuenciales)
scaled_y = scaler_y.fit_transform(datos_y)

print("3. Creando ventanas de tiempo...")
WINDOW_SIZE = 20
FORECAST_SIZE = 5

def create_hybrid_windows(data_seq, data_macro, data_y, window_size, forecast_size):
    X_seq, X_macro, y = [], [], []
    for i in range(len(data_seq) - window_size - forecast_size + 1):
        X_seq.append(data_seq[i:(i + window_size), :])
        X_macro.append(data_macro[i + window_size - 1, :])
        y.append(data_y[(i + window_size):(i + window_size + forecast_size), :])
    return np.array(X_seq), np.array(X_macro), np.array(y)

X_seq, X_macro, y_windows = create_hybrid_windows(scaled_seq, datos_macro, scaled_y, WINDOW_SIZE, FORECAST_SIZE)

n = len(X_seq)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

X_train_seq, X_train_macro, y_train = X_seq[:train_end], X_macro[:train_end], y_windows[:train_end]
X_val_seq, X_val_macro, y_val = X_seq[train_end:val_end], X_macro[train_end:val_end], y_windows[train_end:val_end]
X_test_seq, X_test_macro, y_test = X_seq[val_end:], X_macro[val_end:], y_windows[val_end:]

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

print("4. Construyendo modelo híbrido...")
num_features = X_train_seq.shape[2]
model = build_complex_model(WINDOW_SIZE, num_features, FORECAST_SIZE, 12)

num_params = model.count_params()
print(f"Número de Parámetros del Modelo: {num_params}")

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)

print("\\n--- Iniciando el Entrenamiento Híbrido ---")
history = model.fit(
    [X_train_seq, X_train_macro], y_train,
    validation_data=([X_val_seq, X_val_macro], y_val),
    epochs=100,
    batch_size=32,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

print("5. Realizando predicciones...")
y_train_pred_scaled = model.predict([X_train_seq, X_train_macro])
y_val_pred_scaled = model.predict([X_val_seq, X_val_macro])
y_test_pred_scaled = model.predict([X_test_seq, X_test_macro])

def desnormalizar(pred, shape_orig):
    pred_flat = pred.reshape(-1, 12)
    pred_desnorm = scaler_y.inverse_transform(pred_flat)
    return pred_desnorm.reshape(shape_orig)

y_train_pred = desnormalizar(y_train_pred_scaled, y_train_pred_scaled.shape)
y_train_real = desnormalizar(y_train, y_train.shape)
y_val_pred = desnormalizar(y_val_pred_scaled, y_val_pred_scaled.shape)
y_val_real = desnormalizar(y_val, y_val.shape)
y_test_pred = desnormalizar(y_test_pred_scaled, y_test_pred_scaled.shape)
y_test_real = desnormalizar(y_test, y_test.shape)

def calcular_metricas(y_real, y_pred, subset_name):
    y_real_flat = y_real.flatten()
    y_pred_flat = y_pred.flatten()
    mse = mean_squared_error(y_real_flat, y_pred_flat)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_real_flat, y_pred_flat)
    return float(mse), float(rmse), float(mae)

mse_train, rmse_train, mae_train = calcular_metricas(y_train_real, y_train_pred, "Train")
mse_val, rmse_val, mae_val = calcular_metricas(y_val_real, y_val_pred, "Validation")
mse_test, rmse_test, mae_test = calcular_metricas(y_test_real, y_test_pred, "Test")

def calculate_hit_ratio(y_real, y_pred):
    real_dir = np.sign(y_real.flatten())
    pred_dir = np.sign(y_pred.flatten())
    mask = real_dir != 0
    hits = (real_dir[mask] == pred_dir[mask]).astype(int)
    return float(np.mean(hits))

hit_train = calculate_hit_ratio(y_train_real, y_train_pred)
hit_val = calculate_hit_ratio(y_val_real, y_val_pred)
hit_test = calculate_hit_ratio(y_test_real, y_test_pred)

results_complex = {
    'train_mse': mse_train, 'val_mse': mse_val, 'test_mse': mse_test,
    'train_rmse': rmse_train, 'val_rmse': rmse_val, 'test_rmse': rmse_test,
    'train_mae': mae_train, 'val_mae': mae_val, 'test_mae': mae_test,
    'train_hit': hit_train, 'val_hit': hit_val, 'test_hit': hit_test,
    'params': num_params
}
with open('results_complejo.json', 'w') as f: 
    json.dump(results_complex, f)
print("Finished saving metrics.")
