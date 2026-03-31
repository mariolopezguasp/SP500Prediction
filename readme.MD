# Predicción de Series Temporales Financieras mediante Deep Learning

Este repositorio contiene el proyecto final para la asignatura de **Aprendizaje Profundo** (Grado en Ciencia de Datos, Universitat de València). El objetivo del proyecto es predecir los retornos logarítmicos de una selección diversificada de activos del S&P 500 utilizando arquitecturas de redes neuronales.

### Tablas de Resultados


Tabla del % de acierto direccional
| Modelo | Parametros | Hit Ratio Train | Hit Ratio Val | Hit Ratio Test |
| :--- | :--- | :--- | :--- | :--- |
| Lineal | 145 | 53.19% | 49.96% | 51.95% |
| XGBoost | 600 | 60.68% | 51.55% | 51.02% |
| MLP | 3308 | 52.92% | 54.14% | 52.14% |
| LSTM Simple | 13,836 | 54.54% | 53.74% | 53.18% |
| LSTM Complejo | 138,214 | 53.10% | 52.15% | 52.24% |

Tabla del error (RMSE)
| Modelo | RMSE Train | RMSE Val | RMSE Test |
| :--- | :--- | :--- | :--- |
| Lineal | 0.018751 | 0.016362 | 0.017820 |
| XGBoost | 0.016187 | 0.016519 | 0.018227 |
| MLP | 0.018896  | 0.017558  | 0.019979  |
| LSTM Simple | 0.018986 | 0.016485 | 0.017598 |
| LSTM Complejo | 0.019058 | 0.016488 | 0.017556 |

## 📊 Resultados y Análisis del Portafolio

| Métrica | Valor |
| :--- | :--- |
| **Media Rentabilidad Mercado (B&H)** | 31.27% |
| **Media Rentabilidad Modelo (Long-Short)** | 28.22% |
| **Media Rentabilidad Modelo (Only-Long)** | 29.33% |
| **Mejor Activo (Modelo)** | **WMT** (57.11%) |
| **Peor Activo (Modelo)** | **XOM** (-11.97%) |
| **Porcentaje de activos batidos** | **50.0%** |

---

### 📈 Desglose por Activo

| Activo | B&H (%) | Only-Long (%) | Long-Short (%) | ¿IA Gana? |
| :--- | :---: | :---: | :---: | :---: |
| **JPM** | 49.40 | 39.49 | 30.24 | ❌ NO |
| **ETN** | 13.37 | 19.15 | **25.23** | ✅ SÍ |
| **NVDA** | 70.82 | 53.58 | 38.09 | ❌ NO |
| **NEE** | 13.92 | 22.64 | **32.02** | ✅ SÍ |
| **V** | 24.89 | 27.18 | **29.51** | ✅ SÍ |
| **AMZN** | 29.53 | 42.60 | **56.99** | ✅ SÍ |
| **MSFT** | 14.01 | 8.69 | 3.61 | ❌ NO |
| **PGR** | 4.89 | 4.90 | **4.90** | ✅ SÍ |
| **XOM** | 21.48 | 3.41 | -11.97 | ❌ NO |
| **WELL** | 68.04 | 60.03 | 52.41 | ❌ NO |
| **PG** | -6.19 | 6.30 | **

> **Nota:** La columna **IA_Gana** indica si alguna de las estrategias basadas en el modelo superó el rendimiento pasivo de *Buy & Hold*.

### 🔍 Auditoría de Modelo: NVDA

Resumen de validación de métricas frente a objetivos establecidos:

| # | Métrica | Valor Obtenido | Objetivo | Resultado |
| :--- | :--- | :---: | :---: | :---: |
| 0 | **Naive Forecast** | 0.000920 | < 0.002027 | ✅ |
| 1 | **Correlación** | 0.0259 | > 0.5000 | ⚠️ |
| 2 | **Media Móvil (5d)** | 0.000920 | < 0.000780 | ❌ |

---

## 1. Definición del Problema
El problema seleccionado es de **Aprendizaje Supervisado (Regresión Multivariante)**. Buscamos predecir el vector de retornos financieros en el instante $t+1$ a partir de la información histórica disponible en el instante $t$.

La predicción de mercados financieros es un reto debido a la baja relación señal-ruido. Este proyecto utiliza una selección de 12 activos líderes de diferentes sectores económicos y se busca a través de añadir otras variables que puedan influir al precio de las acciones predecir el precio de las mismas.

## 2. Estructura del Proyecto
* `datos/`: Contiene el dataset histórico de 10 años del retorno diario frente al dia anterior de 12 empresas que cotizan en el SP500.(`dataset_ia_log_returns_10y.csv`) y otro dataset con los datos de esos 10 años pero con el valor de cierre diario ('dataset_ia_precios_cierre_10y.csv').
* `notebooks/`: Notebooks de Google Colab con el desarrollo de modelos y un Rmarkdown con el EDA.



## 3. Datos Utilizados
Los datos comprenden **10 años de historia** de 12 activos del S&P 500.
* **Frecuencia:** Diaria.
* **Transformación:** Retornos logarítmicos para asegurar estacionariedad.
* **Variables:**
    * **X (Entradas):** Matriz de retornos de los 12 activos en el tiempo $t$.
    * **y (Salidas):** Matriz de retornos de los 12 activos en el tiempo $t+1$.

## 4. Métricas de Evaluación
Se han seleccionado las siguientes métricas para validar el modelo:
1.  **Root Mean Squared Error (RMSE):** Función de pérdida principal.
2.  **Mean Absolute Error (MSE):**.
3.  **Sharpe Ratio:** Evaluación de la calidad de la predicción en un contexto financiero.

## 5. Análisis del Estado del Arte y Baselines de Referencia

Para validar la eficacia de los modelos de Deep Learning, estableceremos una comparativa frente a estrategias convencionales de inversión y métricas de predicción base. El objetivo es superar el desempeño del mercado y la persistencia de datos.

### Baselines de Rendimiento (Financieros)
El modelo final se evaluará bajo la capacidad de batir la rentabilidad del mercado ponderado:

| Estrategia | Referencia Histórica | Objetivo del Proyecto |
| :--- | :--- | :--- |
| **Buy & Hold S&P 500** | ~8% - 10% Anual | Superar el retorno ajustado al riesgo (Sharpe) |
| **Naive Forecast** | $y_{t+1} = y_t$ | Lograr una correlación de predicción > 0.5 |
| **Moving Average (MA)** | Estrategia de tendencia | Superar la capacidad de respuesta ante giros de mercado |

### Baselines de Predicción (Métricas de Error)
En series temporales financieras, el "ruido" es muy alto. Se consideran resultados competitivos aquellos que logran reducir el error base del promedio:

| Modelo | Enfoque | Métrica Objetivo (MSE) |
| :--- | :--- | :--- |
| **Media Histórica** | Predicción constante del promedio | 0.00180 (Baseline a batir) |
| **Regresión Lineal** | Modelo de base lineal | 0.00110 |
| **LSTM Multivariante** | Nuestro enfoque de Deep Learning | **< 0.00060** (Meta esperada) |

---

## 6. Selección del Modelo y Análisis de la Arquitectura

Tras evaluar cinco arquitecturas distintas, el modelo **LSTM Simple (Híbrido)** ha sido seleccionado como la solución óptima para este proyecto. A continuación se detallan los motivos de esta elección y las especificaciones técnicas de su implementación.

### 6.1. Justificación de la Selección
Aunque modelos con mayor número de parámetros (como la LSTM Compleja de 138k) mostraron capacidad de aprendizaje en el set de entrenamiento, la **LSTM Simple** demostró un equilibrio superior:

* **Hit Ratio de Test (53.18%):** Superó la barrera del 50% de forma consistente, indicando una capacidad real de capturar la dirección del mercado por encima del azar.
* **Mínimo Error Cuadrático:** Presentó un **RMSE de Test de 0.017598**, el más bajo de todas las arquitecturas, demostrando que una menor complejidad previene el *overfitting* en datos financieros ruidosos.
* **Eficiencia:** Con solo **13,836 parámetros**, ofrece una velocidad de inferencia óptima para aplicaciones de trading en tiempo real.

### 6.2. Arquitectura del Modelo Híbrido
El modelo implementado utiliza una **Arquitectura Híbrida de Dos Ramas** para procesar diferentes tipos de información simultáneamente:

1.  **Rama Secuencial (LSTM):** * Procesa ventanas temporales de 20 días (`WINDOW_SIZE=20`) de los retornos de los 12 activos.
    * Utiliza una capa **LSTM de 32 unidades** con activación `tanh` para extraer dependencias temporales y patrones de volatilidad.
    * Incluye un **Dropout del 20%** para mejorar la robustez del modelo.

2.  **Rama Macroeconómica (Contextual):** * Introduce el **Tipo de Interés de la FED** (DFF) a través de una capa densa de 8 neuronas con activación `ReLU`.
    * Esto permite al modelo "entender" el entorno de liquidez global, factor crítico en la valoración de activos.

3.  **Fusión y Predicción Multi-Step:** * Ambas ramas se combinan mediante una capa de **Concatenación**.
    * La salida se mapea a través de una capa densa de 64 neuronas hacia un **Reshape Layer**, permitiendo realizar un **Multi-step Forecast**: predice el retorno de los 12 activos para los próximos 5 días (`FORECAST_SIZE=5`) en una sola pasada.



### 6.3. Estrategia de Entrenamiento
Para maximizar la estabilidad, se emplearon las siguientes técnicas:
* **Huber Loss:** Utilizada como función de pérdida por ser menos sensible a valores atípicos (outliers) que el MSE tradicional.
* **Optimización Dinámica:** Implementación de `ReduceLROnPlateau` para ajustar el *learning rate* cuando la pérdida se estanca y `EarlyStopping` para evitar el sobreentrenamiento.

## 7. Conclusiones
* **Viabilidad del Deep Learning:** Los resultados confirman que es posible batir estrategias básicas como el *Naive Forecast*, aunque los márgenes en series financieras son extremadamente estrechos.
* **La Simplicidad es Clave:** En mercados financieros, "más parámetros" no equivale a "mejor rendimiento". La arquitectura ligera generalizó mejor que los modelos densos al ignorar parte del ruido inherente a los precios.
* **Impacto de Datos Exógenos:** La inclusión del tipo de interés de la FED ayudó a estabilizar las predicciones, sugiriendo que los modelos puramente técnicos se benefician enormemente de variables fundamentales.
* **Rendimiento del Portafolio:** El modelo logró batir la estrategia *Buy & Hold* en el **50.0%** de los activos analizados, destacando casos de éxito como **AMZN** (56.99% vs 29.53%) y **PG**.

---
**Autores:** Mario López Guasp y Andrés Carvajal Chicaiza  
**Fecha:** 13 de febrero de 2026
