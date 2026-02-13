# Predicción de Series Temporales Financieras mediante Deep Learning

Este repositorio contiene el proyecto final para la asignatura de **Aprendizaje Profundo** (Grado en Ciencia de Datos, Universitat de València). El objetivo del proyecto es predecir los retornos logarítmicos de una selección diversificada de activos del S&P 500 utilizando arquitecturas de redes neuronales.

## 1. Definición del Problema
El problema seleccionado es de **Aprendizaje Supervisado (Regresión Multivariante)**. Buscamos predecir el vector de retornos financieros en el instante $t+1$ a partir de la información histórica disponible en el instante $t$.

La predicción de mercados financieros es un reto debido a la baja relación señal-ruido. Este proyecto utiliza una selección de 12 activos líderes de diferentes sectores económicos, filtrados mediante técnicas de **Hierarchical Risk Parity (HRP)** para maximizar la diversificación.

## 2. Estructura del Proyecto
* `datos/`: Contiene el dataset histórico (`dataset_ia_log_returns_10y.csv`).
* `notebooks/`: Notebooks de Google Colab con el EDA y el desarrollo de modelos.
* `src/`: Código fuente auxiliar y scripts de preprocesamiento.
* `requirements.txt`: Librerías necesarias (pandas, numpy, tensorflow, etc.).

## 3. Datos Utilizados
Los datos comprenden **10 años de historia** de 12 activos del S&P 500.
* **Frecuencia:** Diaria.
* **Transformación:** Retornos logarítmicos para asegurar estacionariedad.
* **Variables:**
    * **X (Entradas):** Matriz de retornos de los 12 activos en el tiempo $t$.
    * **y (Salidas):** Matriz de retornos de los 12 activos en el tiempo $t+1$.

## 4. Métricas de Evaluación
Se han seleccionado las siguientes métricas para validar el modelo:
1.  **Mean Squared Error (MSE):** Función de pérdida principal.
2.  **Mean Absolute Error (MAE):** Medida de error promedio en magnitud.
3.  **Sharpe Ratio:** Evaluación de la calidad de la predicción en un contexto financiero.

## 5. Análisis del Estado del Arte


| Modelo | Técnica | Ventaja | Métrica Ref (MSE) |
| :--- | :--- | :--- | :--- |
| **ARIMA** | Estadística Clásica | Baseline robusto para series lineales | 0.00125 |
| **LSTM** | RNN (Deep Learning) | Captura dependencias temporales largas | 0.00042 |
| **GRU** | RNN (Deep Learning) | Más eficiente que LSTM (menos parámetros) | 0.00045 |
| **CNN-LSTM** | Híbrido | Extracción de patrones locales y temporales | 0.00035 |
| **Transformers** | Attention Mechanisms | Captura de relaciones globales de largo plazo | 0.00031 |

## 6. Ejecución en Google Colab
El notebook en `notebooks/DLFase1.ipynb` incluye:
1.  Carga de datos y **EDA**.
2.  Estadísticas descriptivas y matrices de correlación.
3.  Generación de variables `X` e `y` listas para el entrenamiento.

---
**Autor:** Mario López Guasp
**Fecha:** 13 de febrero de 2026
