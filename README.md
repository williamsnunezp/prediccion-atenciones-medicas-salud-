# 🏥 Predicción de Asistencia a Citas Médicas – Hospital María Auxiliadora

Este proyecto implementa un sistema de **Machine Learning supervisado** para predecir si un paciente asistirá o no a su cita médica en el Hospital de Apoyo María Auxiliadora (Lima, Perú).  
Se sigue la metodología **CRISP-DM** e incluye las fases **1 a 5**, desarrolladas íntegramente en Python.

🔗 **Fuente de datos:**  
Portal de Datos Abiertos del Gobierno del Perú  
https://www.datosabiertos.gob.pe/dataset/citas-medicas-en-el-hospital-de-apoyo-maria-auxiliadora-hma

---

## 🎯 Objetivo del Proyecto

Desarrollar un modelo que prediga la asistencia de pacientes a sus citas médicas utilizando información:

- Administrativa  
- Demográfica  
- Geográfica  
- Temporal  
- De comportamiento (patrones históricos)

Esta predicción sirve para:

- Reducir inasistencias y tiempos muertos.  
- Optimizar disponibilidad de médicos e infraestructura.  
- Mejorar la eficiencia del flujo de atención.  
- Soportar decisiones hospitalarias en planificación diaria y semanal.

---

## ⚙️ Metodología CRISP-DM

| Fase | Descripción |
|------|-------------|
| **1. Comprensión del negocio** | Análisis del problema de inasistencias, impacto operativo y objetivos predictivos. |
| **2. Comprensión de los datos** | Análisis exploratorio, validación de la calidad de datos, distribución de variables y correlaciones. |
| **3. Preparación de datos** | Limpieza, codificación, normalización, ingeniería de características y manejo de desbalance. |
| **4. Modelado** | Entrenamiento de modelos LightGBM y XGBoost con múltiples semillas y validación cruzada. |
| **5. Evaluación** | Comparación de modelos individuales y ensamblajes, análisis detallado de métricas. |
| **6. Despliegue (propuesto)** | Integración futura con sistemas hospitalarios para apoyo en la toma de decisiones. |

---

## 🧪 Modelado Implementado

El proyecto emplea dos modelos de **Gradient Boosting** altamente eficientes para datos tabulares:

### 🔹 LightGBM
- Training acelerado  
- Eficiente con datasets grandes  
- Buen rendimiento con variables categóricas codificadas  

### 🔹 XGBoost
- Robusto en presencia de ruido  
- Regularización L1/L2  
- Convergencia rápida  
- Entrenado con aceleración GPU (CUDA)  

---

## 🎲 Estrategia Especial de Entrenamiento

### ✔ Uso de Múltiples Semillas (42, 123 y 456)
Para garantizar estabilidad y reducir varianza:

- Cada semilla genera particiones independientes  
- Se entrena un modelo LGBM y un XGB por semilla  
- **Total: 6 modelos individuales**

### ✔ Partición de Datos en Dos Niveles
- **Test (no visto): 15%**  
- **Train + Validation: 85%**  
  - Dentro: **85% train**, **15% validación** por semilla

### ✔ Balanceo con SMOTETomek
- Corrige desbalance  
- Mejora el recall de la clase minoritaria  

### ✔ Validación cruzada estratificada (5-fold)
- Aplicada por cada modelo (semilla × algoritmo)  
- Utiliza el número óptimo de árboles obtenido por **early stopping**  

---

## 🧬 Ingeniería de Características

Variables derivadas creadas:

- `Diferencia_dias` (solicitud → cita)
- `mes_cita`
- `semana_mes_cita`
- `bimestre_cita`
- `trimestre_cita`
- `semestre_cita`
- `estacion_cita` (verano/otoño/invierno/primavera)
- `Cita_mes_diferente`

Variables finales incluidas en el modelo:

- Datos demográficos  
- Datos administrativos  
- Datos temporales  
- Variables codificadas y estandarizadas  

---

## 🤖 Ensamblajes de Modelos

Se implementaron dos estrategias:

### 🔸 1. Ensamblaje Ponderado por Semilla (50% LGBM + 50% XGB)
- 3 ensamblajes individuales (uno por semilla)
- 1 ensamblaje final agregado

### 🔸 2. Ensamblaje por Votación Mayoritaria
- **LGBM Ensemble** (promedio de sus 3 semillas)
- **XGB Ensemble** (promedio de sus 3 semillas)
- **Global Ensemble** (50% LGBM Ensemble + 50% XGB Ensemble)

---

## 🏆 Modelo Final Seleccionado

### ⭐ **Global Ensemble (Votación Mayoritaria)**

Seleccionado por:

- **F1 Score más alto:** 0.8173  
- **Accuracy más alto:** 0.8193  
- **ROC AUC:** 0.8864  
- **Recall en clase positiva:** 92.32%  
- Menor varianza entre semillas  
- Mejor robustez general  

Este método combina la fortaleza de:

- **LightGBM →** mayor precisión en clase negativa  
- **XGBoost →** mayor recall y mejor discriminación  

---

## 📊 Resultados Finales

### 🔍 Métricas del mejor ensamblaje (Global Ensemble)

| Métrica | Resultado |
|---------|-----------|
| **Accuracy** | 0.8193 |
| **F1 Weighted** | 0.8173 |
| **ROC AUC** | 0.8864 |
| **Recall clase 1** | 0.9232 |
| **Precision clase 1** | 0.7643 |

---

## 📎 Requisitos del Proyecto

- Python 3.10+  
- Pandas  
- NumPy  
- Scikit-learn  
- LightGBM  
- XGBoost (GPU opcional)  
- Imbalanced-learn  
- Matplotlib / Seaborn  

---

## 📚 Créditos Académicos

Proyecto desarrollado con fines educativos como parte de una investigación en Ciencia de Datos, usando datos abiertos del **Ministerio de Salud del Perú (MINSA)**.

---







