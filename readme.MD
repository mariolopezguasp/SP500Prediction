# Predicción de Series Temporales Financieras con Deep Learning

## 1. Descripción del Proyecto
Este trabajo aborda la predicción del comportamiento de activos del S&P 500 utilizando técnicas de Aprendizaje Profundo (Deep Learning). El objetivo es superar las limitaciones de los modelos lineales tradicionales mediante el uso de redes neuronales recurrentes (LSTM/GRU) capaces de capturar dependencias temporales complejas y no lineales.

### Objetivos Específicos
1.  **Minimización de Incertidumbre:** Reducir el Error Cuadrático Medio (MSE) en la predicción de retornos.
2.  **Clasificación Direccional:** Obtener un Accuracy > 50% en la predicción del signo del movimiento (subida/bajada).
3.  **Integración Multivariante:** Explotar la información conjunta de Precio, Volatilidad (High-Low) y Volumen.

## 2. Datos y Metodología
- **Dataset:** S&P 500 Completo (últimos 10 años).
- **Formato:** OHLCV (Open, High, Low, Close, Volume).
- **Variable Objetivo ($y$):** Retorno Logarítmico Diario ($R_t = \ln(P_t / P_{t-1})$). Se usa retorno logarítmico por sus propiedades de estacionariedad, simetría y aditividad.
- **Ventana Temporal:** 30 días (Trading Month).

### Estructura de Tensores
El notebook `01_Seleccion_Datos_y_EDA.ipynb` transforma los datos crudos en tensores listos para el entrenamiento:
- **X (Entrada):** `(N_muestras, 30, 5)`
- **y (Salida):** `(N_muestras, )`

## 3. Estado del Arte y Baselines

La validación del modelo se realizará comparando su rendimiento contra los siguientes estándares de la industria:

| Modelo | Tipo | Descripción | Métrica Clave |
| :--- | :--- | :--- | :--- |
| **Naive Forecast** | Baseline | Asume que el mercado es un paseo aleatorio ($y_{t+1} = y_t$). | MSE |
| **ARIMA** | Estadístico | Modelo lineal clásico. | AIC / RMSE |
| **LSTM** | Deep Learning | Red Neuronal Recurrente con memoria a largo plazo. Estándar actual. | Directional Accuracy |
| **Transformer** | Deep Learning | Arquitectura basada en Atención (SOTA). | Sharpe Ratio |

## 4. Estructura del Repositorio
```text
├── data/               # Datos procesados (no incluidos en repo por tamaño)
├── notebooks/          # 01_Seleccion_Datos_y_EDA.ipynb
├── requirements.txt    # Dependencias
└── README.md           # Este informe# SP500Prediction
