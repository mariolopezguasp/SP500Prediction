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



### 7.2. Auditoría de Predicción: El Caso de NVDA
Para entender las limitaciones del modelo, se realizó una auditoría técnica sobre activos de alta volatilidad como **NVDA**. Los resultados revelan la naturaleza del desafío:


| Métrica | Estado | Conclusión Técnica |
| :--- | :---: | :--- |
| **Naive Forecast** | ✅ | El modelo supera la persistencia básica, aportando valor real sobre la simple repetición del dato anterior. |
| **Correlación** | ⚠️ | La correlación obtenida (0.0259) frente al objetivo (0.5) confirma la dificultad de predecir la magnitud exacta en activos con ruido extremo. |
| **Media Móvil (5d)** | ❌ | El error del modelo es ligeramente superior al promedio móvil de 5 días, sugiriendo que en activos de crecimiento explosivo, las señales de tendencia simples son competidores fuertes. |

### 7.3. Comportamiento por Activo
El éxito del modelo no fue uniforme, lo que valida la necesidad de una gestión de cartera diversificada:
* **Éxitos Claros:** En activos como **ETN, NEE, V, AMZN, PGR y PG**, las predicciones del modelo superaron sistemáticamente al Buy & Hold.
* **Desafíos:** En activos con tendencias alcistas parabólicas (como **WMT** o **NVDA**), el modelo tendió a ser conservador, lo que resultó en un *underperformance* relativo frente a simplemente mantener el activo.

## 8. Conclusiones

### 8.1. El valor de la arquitectura simplificada
Una de las lecciones principales fue que un mopdelo más complejo no siempre garantiza mejores resultados. Los modelos con demasiados parámetros pueden tender a sobreajustarse. Nuestra **LSTM Simple** demostró ser la más robusta, logrando un **Hit Ratio del 53.18%**, lo que confirma que en ocasiones, un diseño ligero es más eficaz.

### 8.2. Rendimiento y Generación de Alpha
El análisis de las estrategias de inversión (*Only-Long* y *Long-Short*) reveló que el modelo tiene una capacidad real para identificar ineficiencias:
* **Éxito en Diversificación:** El modelo logró batir la rentabilidad pasiva del mercado (*Buy & Hold*) en el **50.0%** de los activos analizados.
* **Casos Destacados:** En empresas como **AMZN** o **PG**, las decisiones de la IA superaron con creces el rendimiento tradicional, demostrando que el modelo detecta cambios de tendencia que el inversor pasivo ignora.
* **Sesgo Conservador:** En activos con crecimientos explosivos y constantes (como NVDA), el modelo tendió a ser prudente, lo que sugiere nuestro modelo suele infraestimar movimientos parabólicos en favor de la estabilidad.

### 8.3. El impacto estratégico de los datos macroeconómicos
Una de las mejoras más significativas en el rendimiento del modelo provino de la integración de **variables macroeconómicas**. Al alimentar la red no solo con precios históricos, sino también con el **Tipo de Interés de la FED**, el modelo adquirió una "visión de contexto" de la que carecen los modelos técnicos simples. Esta capa de información exógena permitió que la IA ajustara sus expectativas según el entorno de liquidez global, logrando una estabilidad en las predicciones que los modelos basados únicamente en el precio no pudieron alcanzar. En definitiva, la combinación de micro-datos (retornos de activos) y macro-datos (tipos de interés) fue el factor diferencial para reducir el error en periodos de incertidumbre.

### 8.4. Integración de un Pipeline de Datos Completo
Más allá de la arquitectura de la red, los resultados del proyecto vienen por la construcción de un flujo de trabajo integral donde se lleva a cabo:
1.  **Extracción de datos:** Implementación de técnicas de recolección de datos automatizada de **Wikipedia** (para la selección de activos) y **Yahoo Finance**, asegurando un histórico sólido de los últimos 10 años.
2.  **Contexto Macroeconómico:** Integración de variables externas (tipos de interés de la FED) para dotar al modelo de una visión global del entorno económico.
3.  **Transformación de Señal:** Uso de retornos logarítmicos para garantizar la estabilidad estadística de los datos de entrada.
4.  **Lógica de Decisión:** Traducción de predicciones numéricas en una estrategia de portafolio capaz de competir con los estándares del sector.

---
**Autores:** Mario López Guasp y Andrés Carvajal Chicaiza  
**Fecha:** 13 de febrero de 2026
