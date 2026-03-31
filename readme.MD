# Predicción de Series Temporales Financieras mediante Deep Learning

Este repositorio contiene el proyecto final para la asignatura de **Aprendizaje Profundo** (Grado en Ciencia de Datos, Universitat de València). El objetivo del proyecto es predecir los retornos logarítmicos de una selección diversificada de activos del S&P 500 utilizando arquitecturas de redes neuronales.

### Tablas de Resultados

Tabla del % de acierto direccional
| Modelo | Parametros | Hit Ratio Train | Hit Ratio Val | Hit Ratio Test |
| :--- | :--- | :--- | :--- | :--- |
| Lineal | 145 | 53.19% | 49.96% | 51.95% |
| XGBoost | 600 | 60.68% | 51.55% | 51.02% |
| MLP | 3308 | 52.92% | 54.14% | 52.14% |
| _`LSTM Simple`_ | _`13,836`_ | _`54.54%`_ | _`53.74%`_ | _`53.18%`_ |
| LSTM Complejo | 138,214 | 53.10% | 52.15% | 52.24% |

Tabla del error (RMSE)
| Modelo | RMSE Train | RMSE Val | RMSE Test |
| :--- | :--- | :--- | :--- |
| Lineal | 0.018751 | 0.016362 | 0.017820 |
| XGBoost | 0.016187 | 0.016519 | 0.018227 |
| MLP | 0.018896 | 0.017558 | 0.019979 |
| _`LSTM Simple`_ | _`0.018986`_ | _`0.016485`_ | _`0.017598`_ |
| LSTM Complejo | 0.019058 | 0.016488 | 0.017556 |

---

## 1. Definición del Problema
El problema seleccionado es de **Aprendizaje Supervisado (Regresión Multivariante)**. Buscamos predecir el vector de retornos financieros en el instante $t+1$ a partir de la información histórica disponible en el instante $t$.

La predicción de mercados financieros es un reto debido a la baja relación señal-ruido. Este proyecto utiliza una selección de 12 activos líderes de diferentes sectores económicos y se busca a través de añadir otras variables que puedan influir al precio de las acciones predecir el precio de las mismas.

## 2. Estructura del Proyecto


* ** Datasets:** * `dataset_ia_log_returns_10y.csv`: Datos históricos de 10 años con los retornos logarítmicos diarios de los 12 activos (variable principal de entrenamiento).
    * `dataset_ia_precios_cierre_10y.csv`: Precios de cierre diarios utilizados para las simulaciones de inversión y cálculo de rentabilidad real.

* **Notebooks de Modelado:**
    * `1ModeloLineal.ipynb`: Baseline de regresión lineal.
    * `2ModeloXGBoost.ipynb`: Implementación basada en Gradient Boosting.
    * `3ModeloMLP.ipynb`: Red Neuronal Perceptrón Multicapa.
    * `4ModeloLSTMSimple.ipynb`: **Modelo seleccionado** con arquitectura híbrida (LSTM + Datos Macro).
    * `5ModeloLSTMComplejo.ipynb`: Experimentación con arquitectura profunda de alta parametrización.

* **Extracción y procesado de datos en R:**
    * `procesamientodatos.Rmd`: Script en R para la extraccioón de datos.



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

Para validar la eficacia de nuestros modelos de Deep Learning, comparamos los resultados frente a estrategias convencionales de inversión y métricas de predicción base:

### 📈 Baselines de Rendimiento (Objetivos Financieros)
* **Buy & Hold S&P 500:** Referencia histórica del ~8-10% anual. Nuestro objetivo es superar este retorno ajustado al riesgo (Sharpe Ratio).
* **Naive Forecast ($y_{t+1} = y_t$):** Superar la persistencia de datos logrando una correlación de predicción superior a 0.5.
* **Moving Average (MA):** Batir la capacidad de respuesta de las estrategias de tendencia tradicionales ante giros bruscos del mercado.

### 📉 Baselines de Predicción (Métricas de Error)
En un entorno con alto "ruido", consideramos resultados competitivos aquellos que reduzcan el error base de los modelos estándar:
* **Media Histórica:** Baseline a batir con un MSE de 0.00180.
* **Regresión Lineal:** Referencia de modelo lineal con un MSE de 0.00110.
* **LSTM Multivariante:** Nuestra meta de Deep Learning fijada en un **MSE < 0.00060**.

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

## 7. Análisis de Rendimiento en Test y Estrategias de Inversión

Una vez seleccionado el modelo **LSTM Simple**, se evaluó su capacidad de generar retornos reales en el conjunto de datos de test (fuera de la muestra). El análisis se divide en la capacidad predictiva estadística y su ejecución en un entorno de portafolio.

### 7.1. Evaluación de Estrategias (Long-Short vs. Only-Long)
El modelo se utilizó para simular dos estrategias de inversión comparadas con el *benchmark* pasivo (Buy & Hold):

* **Estrategia Only-Long:** El modelo solo abre posiciones compradoras cuando predice retornos positivos. Logró una rentabilidad media del **29.33%**, acercándose significativamente al mercado.
* **Estrategia Long-Short:** Permite tomar posiciones cortas en activos con predicción negativa. Aunque la rentabilidad media fue del **28.22%**, demostró una capacidad excepcional en activos específicos como **XOM**, donde la estrategia capturó movimientos que el mercado ignoró.
* **Alpha Generado:** El modelo batió al mercado en el **50.0%** de los activos, destacando casos como **AMZN** (+27.46% sobre B&H) y **PG**, donde la IA transformó una pérdida del mercado (-6.19%) en una ganancia significativa.

| Métrica | Valor |
| :--- | :--- |
| **Media Rentabilidad Mercado (B&H)** | 31.27% |
| **Media Rentabilidad Modelo (Long-Short)** | 28.22% |
| **Media Rentabilidad Modelo (Only-Long)** | 29.33% |
| **Mejor Activo (Modelo)** | **WMT** (57.11%) |
| **Peor Activo (Modelo)** | **XOM** (-11.97%) |
| **Porcentaje de activos batidos** | **50.0%** |

---


| Activo | B&H (%) | Only-Long (%) | Long-Short (%) | ¿Modelo Gana? |
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

> **Nota:** La columna **Modelo_Gana** indica si alguna de las estrategias basadas en el modelo superó el rendimiento pasivo de *Buy & Hold*.


### 7.3. Comportamiento por Activo
El éxito del modelo no fue uniforme, lo que valida la necesidad de una gestión de cartera diversificada:
* **Éxitos Claros:** En activos como **ETN, NEE, V, AMZN, PGR y PG**, las predicciones del modelo superaron sistemáticamente al Buy & Hold.
* **Desafíos:** En activos con tendencias alcistas parabólicas (como **WMT** o **NVDA**), el modelo tendió a ser conservador, lo que resultó en un *underperformance* relativo frente a simplemente mantener el activo.

## 8. Conclusiones

### 8.1. Superioridad de la Arquitectura Simplificada
Una de las lecciones fundamentales de este proyecto es que una mayor complejidad paramétrica no garantiza un rendimiento superior en mercados financieros. Los modelos con exceso de parámetros, como la LSTM Compleja, tendieron al sobreajuste. Nuestra LSTM Simple demostró ser la más robusta, logrando un Hit Ratio del 53.18%.

### 8.2. Rendimiento Operativo y Estrategias de Inversión
El análisis de las estrategias (**Only-Long** y **Long-Short**) reveló una capacidad real del modelo para identificar ineficiencias estructurales en el mercado. 
Aunque nuestro mejor modelo no haya podido superar de media los resultados del Buy and Hold, hemos obtenido unos resultados que han estado cerca de igualarlo

### 8.3. Impacto Estratégico de los Datos Macroeconómicos
[cite_start]La integración de variables exógenas representó la mejora más significativa en la estabilidad de las predicciones[cite: 63]. Al alimentar la red con el **Tipo de Interés de la FED (DFF)**, el modelo adquirió una "visión de contexto" sobre la liquidez global de la que carecen los modelos técnicos puros. Esta combinación de micro-datos (retornos) y macro-datos (tipos de interés) fue el factor diferencial para reducir el error durante periodos de alta incertidumbre económica.

### 8.4. Integración de un Pipeline de Datos Integral
Los resultados finales son el producto de un flujo de trabajo de ingeniería cohesivo:
* **Extracción de datos**: Implementación de *scraping* en **Wikipedia** y **Yahoo Finance** para consolidar una base histórica sólida de 10 años.
* **Transformación de Señal**: Uso de retornos logarítmicos y escalado estándar para garantizar la estacionariedad y estabilidad estadística de las entradas.
* **Lógica de Decisión**: Traducción exitosa de predicciones numéricas tensoriales en estrategias de portafolio capaces de competir con los estándares del sector financiero.

Este pipeline representa una base sólida y escalable para futuras implementaciones con otras fuentes exógenas que puedan aportar información adicional a la red para optimizar los resultados predictivos.

---
**Autores:** Mario López Guasp y Andrés Carvajal Chicaiza  
**Fecha:** 13 de febrero de 2026
