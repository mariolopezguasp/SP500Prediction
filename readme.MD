# Predicción de Series Temporales Financieras mediante Deep Learning

Este repositorio contiene el proyecto final para la asignatura de **Aprendizaje Profundo** (Grado en Ciencia de Datos, Universitat de València). El objetivo del proyecto es predecir los retornos logarítmicos de una selección diversificada de activos del S&P 500 utilizando arquitecturas de redes neuronales.

## 1. Definición del Problema
El problema seleccionado es de **Aprendizaje Supervisado (Regresión Multivariante)**. Buscamos predecir el vector de retornos financieros en el instante $t+1$ a partir de la información histórica disponible en el instante $t$.

La predicción de mercados financieros es un reto debido a la baja relación señal-ruido. Este proyecto utiliza una selección de 12 activos líderes de diferentes sectores económicos y se busca a través de añadir otras variables que puedan influir al precio de las acciones predecir el precio de las mismas.

## 2. Estructura del Proyecto
* `datos/`: Contiene el dataset histórico de 10 años del retorno diario frente al dia anterior de 12 empresas que cotizan en el SP500.(`dataset_ia_log_returns_10y.csv`).
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

*Nota: Batir la rentabilidad media anual del S&P 500 (aprox. 10%) con una selección de solo 12 activos requiere que el modelo optimice no solo la dirección (precio), sino también la ponderación del riesgo mediante la predicción de la volatilidad.*


---
**Autor:** Mario López Guasp y Andrés Carvajal Chicaiza
**Fecha:** 13 de febrero de 2026
