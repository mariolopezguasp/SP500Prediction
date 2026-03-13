import nbformat as nbf

nb = nbf.v4.new_notebook()

# Metadata
nb.metadata = {
  "colab": {
    "provenance": []
  },
  "kernelspec": {
    "name": "python3",
    "display_name": "Python 3"
  },
  "language_info": {
    "name": "python"
  }
}

cells = []

# Cell 1: imports and data load
cells.append(nbf.v4.new_markdown_cell("## 1. Carga de Datos y Preprocesamiento"))

code_data = '''import sys
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader.data as web
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score

# Añadir el path para importar el modelo
sys.path.append(os.path.abspath('..'))

from modelos.modelo_complejo import build_complex_model

print("1. Cargando datos...")
# Load the dataset
df_ret = pd.read_csv('/content/dataset_ia_log_returns_10y.csv', index_col=0, parse_dates=True)

# Calculamos Volatilidad (Multiparámetro)
df_vol = df_ret.rolling(window=21).std()

# Descargamos los tipos de la FED
fecha_inicio = df_ret.index.min()
fecha_fin = df_ret.index.max()
df_fed = web.DataReader('DFF', 'fred', fecha_inicio, fecha_fin)
df_fed.rename(columns={'DFF': 'Tipo_Interes_FED'}, inplace=True)
df_fed['Tipo_Interes_FED'] = df_fed['Tipo_Interes_FED'] / 100 # Formato decimal

# Unimos todo en un solo dataframe para asegurar que las fechas coinciden
df_master = df_ret.join(df_vol, rsuffix='_vol').join(df_fed, how='left')
df_master['Tipo_Interes_FED'] = df_master['Tipo_Interes_FED'].ffill() # Rellenar nulos de la FED
df_master = df_master.dropna() # Quitamos los 21 primeros días por la volatilidad

# Separar en Secuencial, Macro y Target (Y)
datos_secuenciales = df_master.drop(columns=['Tipo_Interes_FED']).values # Retornos + Volatilidades (24 cols)
datos_macro = df_master[['Tipo_Interes_FED']].values                     # FED (1 col)
datos_y = df_master.iloc[:, :12].values                                  # Solo los 12 Retornos para predecir
'''
cells.append(nbf.v4.new_code_cell(code_data))

# Cell 2: Scaling and Windowing
cells.append(nbf.v4.new_markdown_cell("## 2. Escalado y Creación de Ventanas (Train, Val, Test)"))

code_scale = '''print("2. Escalando datos...")
scaler_seq = MinMaxScaler(feature_range=(-1, 1))
scaler_y = MinMaxScaler(feature_range=(-1, 1))

scaled_seq = scaler_seq.fit_transform(datos_secuenciales)
scaled_y = scaler_y.fit_transform(datos_y)
# La FED ya está en escala [0, ~0.10], la dejamos tal cual para no perder su significado real.

print("3. Creando ventanas de tiempo...")
WINDOW_SIZE = 20
FORECAST_SIZE = 5

def create_hybrid_windows(data_seq, data_macro, data_y, window_size, forecast_size):
    X_seq, X_macro, y = [], [], []
    for i in range(len(data_seq) - window_size - forecast_size + 1):
        # 1. Secuencia de días
        X_seq.append(data_seq[i:(i + window_size), :])
        # 2. Dato macro del DÍA ACTUAL (el último día de la ventana)
        X_macro.append(data_macro[i + window_size - 1, :])
        # 3. Lo que queremos predecir (N días futuros)
        y.append(data_y[(i + window_size):(i + window_size + forecast_size), :])
    return np.array(X_seq), np.array(X_macro), np.array(y)

X_seq, X_macro, y_windows = create_hybrid_windows(scaled_seq, datos_macro, scaled_y, WINDOW_SIZE, FORECAST_SIZE)

# Split 70% Train, 15% Val, 15% Test
n = len(X_seq)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

X_train_seq, X_train_macro, y_train = X_seq[:train_end], X_macro[:train_end], y_windows[:train_end]
X_val_seq, X_val_macro, y_val = X_seq[train_end:val_end], X_macro[train_end:val_end], y_windows[train_end:val_end]
X_test_seq, X_test_macro, y_test = X_seq[val_end:], X_macro[val_end:], y_windows[val_end:]

print(f" Forma X_train_seq (LSTM): {X_train_seq.shape}")
print(f" Forma X_val_seq (LSTM): {X_val_seq.shape}")
print(f" Forma X_test_seq (LSTM): {X_test_seq.shape}")
'''
cells.append(nbf.v4.new_code_cell(code_scale))


# Cell 3: Build Model
cells.append(nbf.v4.new_markdown_cell("## 3. Construcción y Entrenamiento del Modelo Híbrido"))

code_train = '''from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

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

# Show converging learning curves
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Train Loss (Huber)')
plt.plot(history.history['val_loss'], label='Validation Loss (Huber)')
plt.title('Curvas de Aprendizaje - Modelo Híbrido Complejo')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()
'''
cells.append(nbf.v4.new_code_cell(code_train))


# Cell 4: Evaluation
cells.append(nbf.v4.new_markdown_cell("## 4. Evaluación del Modelo"))

code_eval = '''# Predicciones
print("5. Realizando predicciones...")
y_train_pred_scaled = model.predict([X_train_seq, X_train_macro])
y_val_pred_scaled = model.predict([X_val_seq, X_val_macro])
y_test_pred_scaled = model.predict([X_test_seq, X_test_macro])

# Desnormalizar (la métrica RMSE y MAE debe ser en la escala original)
# Como la salida es (N, FORECAST_SIZE, 12), aplanamos temporalmente para desnormalizar
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

# Calcular métricas globales (sobre todos los activos y todos los días futuros)
def calcular_metricas(y_real, y_pred, subset_name):
    y_real_flat = y_real.flatten()
    y_pred_flat = y_pred.flatten()
    
    mse = mean_squared_error(y_real_flat, y_pred_flat)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_real_flat, y_pred_flat)
    
    print(f"--- {subset_name} ---")
    print(f"MSE:  {mse:.6f}")
    print(f"RMSE: {rmse:.6f} ({rmse*100:.2f}%)")
    print(f"MAE:  {mae:.6f} ({mae*100:.2f}%)")
    
    return mse, rmse, mae

mse_train, rmse_train, mae_train = calcular_metricas(y_train_real, y_train_pred, "Train")
mse_val, rmse_val, mae_val = calcular_metricas(y_val_real, y_val_pred, "Validation")
mse_test, rmse_test, mae_test = calcular_metricas(y_test_real, y_test_pred, "Test")

results_complex = {
    'train_mse': mse_train, 'val_mse': mse_val, 'test_mse': mse_test,
    'train_rmse': rmse_train, 'val_rmse': rmse_val, 'test_rmse': rmse_test,
    'train_mae': mae_train, 'val_mae': mae_val, 'test_mae': mae_test,
    'params': num_params
}
with open('results_complejo.json', 'w') as f: 
    json.dump(results_complex, f)
'''
cells.append(nbf.v4.new_code_cell(code_eval))


# Cell 5: Hit Ratio
cells.append(nbf.v4.new_markdown_cell("## 5. Análisis del Hit Ratio (Acierto Direccional)"))

code_hit = '''def calculate_hit_ratio(y_real, y_pred):
    # Flatten y comparar la dirección: si ambos son positivos o ambos negativos, es hit
    real_dir = np.sign(y_real.flatten())
    pred_dir = np.sign(y_pred.flatten())
    # Ignoramos donde real_dir es exactamente 0 (poco probable en floats, pero seguro es seguro)
    mask = real_dir != 0
    hits = (real_dir[mask] == pred_dir[mask]).astype(int)
    return np.mean(hits)

hit_train = calculate_hit_ratio(y_train_real, y_train_pred)
hit_val = calculate_hit_ratio(y_val_real, y_val_pred)
hit_test = calculate_hit_ratio(y_test_real, y_test_pred)

print(f"Hit Ratio Train: {hit_train*100:.2f}%")
print(f"Hit Ratio Val: {hit_val*100:.2f}%")
print(f"Hit Ratio Test: {hit_test*100:.2f}%")

# Gráfico
plt.figure()
x = ['Train', 'Val', 'Test']
y_hits = [hit_train*100, hit_val*100, hit_test*100]
plt.bar(x, y_hits, color=['blue', 'orange', 'green'])
plt.axhline(50, color='red', linestyle='--', label='50% Aleatorio')
plt.title('Hit Ratio Direccional - Modelo Híbrido')
plt.ylabel('Acierto (%)')
plt.legend()
plt.show()
'''
cells.append(nbf.v4.new_code_cell(code_hit))

# Cell 6: Bias-Variance Tradeoff Graph (Técnica de la Tijera)
cells.append(nbf.v4.new_markdown_cell("## 6. Análisis Final: Trade-Off Sesgo-Varianza (Técnica de la Tijera)"))

code_bias_variance = '''import numpy as np
import matplotlib.pyplot as plt

# 1. Definir el eje X continuo (Escala logarítmica de Complejidad x Parámetros)
x_smooth = np.linspace(0.8, 6.0, 300)

# 2. Curvas teóricas para modelos SIN regularizar (según Valero)
bias_curve = 0.000380 * np.exp(-0.8 * (x_smooth - 0.5)) + 0.000250
variance_curve = 0.000001 * np.exp(1.05 * (x_smooth - 0.5))
total_error_curve = bias_curve + variance_curve

# 3. Modelos (Posición X basada en log10 del nº de parámetros reales)
# Reg. Lineal (12 weights) -> log10(12) = 1.08
# Neuronal Simple (156 param) -> log10(156) = 2.19
# XGBoost (~9,000 ramas/decisiones) -> log10(9000) = 3.95
# Complejo Híbrido (341,524 param) -> log10(341524) = 5.53
x_models = np.array([1.08, 2.19, 3.95, 5.53])
model_names = ['Regresión Lineal\\n(12 param)', 'Neuronal Simple\\n(156 param)', 'XGBoost\\n(≈9k split)', 'Complejo Híbrido\\n(341k param)']

# Errores reales extraídos
val_mse_real = np.array([0.000268, 0.000275, 0.000273, 0.000272])
train_mse_real = np.array([0.000352, 0.000349, 0.000262, 0.000363])

fig, ax = plt.subplots(figsize=(12, 7))

# Dibujar curvas teóricas de la receta pura sin "Tijera"
ax.plot(x_smooth, bias_curve, 'b--', linewidth=2, alpha=0.5, label='Sesgo Teórico (Underfitting)')
ax.plot(x_smooth, variance_curve, 'r--', linewidth=2, alpha=0.5, label='Varianza Teórica (Overfitting)')
ax.plot(x_smooth, total_error_curve, 'k-', linewidth=2.5, alpha=0.3, label='Error Validación Teórico (Sin Regularizar)')

# Plotear puntos reales
colors = ['purple', 'cyan', 'orange', 'green']
for i in range(len(x_models)):
    # Validación
    ax.scatter(x_models[i], val_mse_real[i], color=colors[i], s=200, zorder=6, edgecolors='black')
    # Entrenamiento (cruces)
    ax.scatter(x_models[i], train_mse_real[i], color=colors[i], marker='X', s=150, zorder=5, edgecolors='black', alpha=0.7)
    
    texto_resumen = f"{model_names[i]}\\nTrain: {train_mse_real[i]:.6f}\\nVal: {val_mse_real[i]:.6f}"
    
    # Colocar la etiqueta según el modelo
    if i == 3: # Híbrido al bajar la curva
        ax.annotate(texto_resumen, (x_models[i], val_mse_real[i]), textcoords="offset points", xytext=(0,-60), ha='center', fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=colors[i], lw=2, alpha=0.9))
    else:
        ax.annotate(texto_resumen, (x_models[i], max(val_mse_real[i], train_mse_real[i])), textcoords="offset points", xytext=(0,20), ha='center', fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=colors[i], lw=2, alpha=0.9))

# Flecha explicativa: Efecto de la Regularización (Tijera)
y_teorico_complejo = total_error_curve[np.abs(x_smooth - x_models[3]).argmin()]
ax.annotate('', xy=(x_models[3], val_mse_real[3]+0.000005), xytext=(x_models[3], y_teorico_complejo),
            arrowprops=dict(facecolor='green', shrink=0.01, width=2, headwidth=8), zorder=4)
ax.text(x_models[3]-0.15, (y_teorico_complejo + val_mse_real[3])/2, 'Efecto L2 / Dropout\\n(Reduce Varianza Mantenida)', 
        color='green', fontsize=10, ha='right', va='center', fontweight='bold', bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.7))

# Flecha explicativa: Overfitting en XGBoost
ax.annotate('', xy=(x_models[2], train_mse_real[2]+0.000005), xytext=(x_models[2], val_mse_real[2]-0.000005),
            arrowprops=dict(facecolor='red', shrink=0.01, width=2, headwidth=8), zorder=4)
ax.text(x_models[2]+0.15, (val_mse_real[2] + train_mse_real[2])/2, 'Sobreajuste\\n(Brecha Train-Val)', 
        color='red', fontsize=10, ha='left', va='center', fontweight='bold', bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.7))

# Estética y Etiquetas
ax.set_title('Reducción de Varianza por Regularización (Método de Valero)', fontsize=16, fontweight='bold')
ax.set_xlabel(r'Complejidad: $Log_{10}$ (Número de Parámetros) $\\rightarrow$', fontsize=13)
ax.set_ylabel(r'Error Objetivo (MSE) $\\rightarrow$', fontsize=13)
ax.set_xticks(x_models)
ax.set_xticklabels(['$10^1$', '$10^2$', '$10^4$', '$10^{5.5}$'])
ax.grid(True, linestyle=':', alpha=0.6)

# Custom Legend para explicar las marcas
import matplotlib.lines as mlines
val_marker = mlines.Line2D([], [], color='black', marker='o', linestyle='None', markersize=10, label='MSE Validación (Círculo)')
train_marker = mlines.Line2D([], [], color='black', marker='X', linestyle='None', markersize=9, label='MSE Entrenamiento (Cruz)')
handles, labels = ax.get_legend_handles_labels()
handles.extend([val_marker, train_marker])
ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=11)

plt.tight_layout()
plt.show()

print("="*90)
print(" ANÁLISIS DEL EXPERIMENTO SEGÚN LA METODOLOGÍA DE VALERO ".center(90))
print("="*90)
print("1. DEFINICIÓN DE COMPLEJIDAD Y POSICIÓN (EJE X):")
print("   Tal y como se define en la receta de entrenamiento, la complejidad se evalúa por el número")
print("   bruto de parámetros libres o estimaciones. El modelo Lineal (~12) se ubica a la izquierda")
print("   y la Red Compleja (341k parámetros entrenables) es la frontera de extrema derecha.")
print("\\n2. EL PROBLEMA DE LA VARIANZA PURA (XGBoost):")
print("   El modelo de 600 árboles (miles de split nodes o pesos equivalentes) evidencia el concepto de")
print("   overfitting clásico. Su asombroso Error de Train (cruz) cae bruscamente a 0.000262 gracias")
print("   a su flexibilidad, pero al enfrentar Validación (círculo) rebota a 0.000273. (Flecha Roja)")
print("\\n3. EL PODER DE LA 'TIJERA' EN LA RED COMPLEJA (Overparameterized Regularizada):")
print("   Ubicados en el extremo dominante de complejidad, la teoría indica que el error de validación")
print("   debería dispararse (línea negra transparente) al memorizar el dataset. Sin embargo, aplicando")
print("   la regla mágica de 'putear a la red' (Dropout selectivo y Regularización Penalty L2), logramos")
print("   'cortar' esa Varianza drásticamente mediante ruido regulador, demostrando el descenso (Flecha Verde).")
print("   Mantenemos un modelo inmenso capaz de entender problemas macroeconómicos (Sesgo estructural bajo)")
print("   cuyo Overfitting es contrarrestado artificialmente, acercando Validación y Train al óptimo.")
print("="*90)
'''
cells.append(nbf.v4.new_code_cell(code_bias_variance))

nb.cells = cells

# Guardar el notebook
with open('c:/Users/mario/OneDrive/Desktop/uni/4/Proyecto/SP500Prediction/4ModeloComplejo.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
