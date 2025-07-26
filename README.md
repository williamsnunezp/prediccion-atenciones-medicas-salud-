# 🏥 Predicción de Atenciones Médicas en el Hospital María Auxiliadora

Este proyecto aplica técnicas de **Machine Learning supervisado** para predecir la **asistencia de pacientes** a sus citas médicas en el Hospital de Apoyo María Auxiliadora (Lima, Perú). Se sigue la metodología **CRISP-DM**, implementando principalmente las fases **2 a 5**:  
**Comprensión de los datos, Preparación, Modelado y Evaluación.**

🔗 **Fuente de datos oficial**: [Portal de Datos Abiertos del Gobierno del Perú](https://datosabiertos.gob.pe/group/hospital-mar%C3%ADa-auxiliadora?sort_by=changed&f%5B0%5D=changed%3A2024-05-02)

---

## 🎯 Objetivo del Proyecto

Predecir si un paciente asistirá o no a su cita médica programada, utilizando variables clínicas, administrativas y temporales. Esta predicción puede ser útil para:

- Optimizar el uso de recursos hospitalarios (médicos, insumos, salas).
- Reducir ausencias y tiempos de espera.
- Apoyar la planificación del flujo de atención médica.

---

## ⚙️ Metodología CRISP-DM aplicada

| Fase                     | Detalle                                                                 |
|--------------------------|-------------------------------------------------------------------------|
| 1. Comprensión del negocio | Análisis del problema hospitalario y definición de objetivos predictivos. |
| 2. Comprensión de los datos | Revisión del dataset, análisis exploratorio, identificación de patrones.  |
| 3. Preparación de los datos | Limpieza, transformación, codificación de variables y creación de nuevas features. |
| 4. Modelado              | Entrenamiento de modelos con validación cruzada estratificada.          |
| 5. Evaluación            | Métricas de rendimiento, matriz de confusión y selección del mejor modelo. |
| 6. Despliegue (propuesta)| Integración del modelo como apoyo a la toma de decisiones hospitalarias. |

---

## 🧪 Modelos Evaluados

Se probaron cuatro algoritmos principales:

- ✅ LightGBM
- ✅ XGBoost
- ✅ Random Forest
- ✅ Extra Trees

> Se aplicó balanceo de clases con **SMOTETomek** y normalización si fue necesario.

---

## 📊 Variables Relevantes

- `Edad`
- `Tipo de seguro`
- `Modalidad de cita` (presencial o remoto)
- `Especialidad médica`
- `Diferencia de días entre solicitud y cita`
- `Estación`, `bimestre`, `semana del mes` (derivadas de fecha)

---

## 📈 Resultados Globales

| Modelo         | Accuracy | F1 Score | ROC AUC |
|----------------|----------|----------|---------|
| LightGBM       | 0.81     | 0.81     | 0.88    |
| XGBoost        | 0.76     | 0.76     | 0.84    |
| Random Forest  | 0.78     | 0.78     | 0.86    |
| Extra Trees    | 0.77     | 0.77     | 0.83    |

> Se usó validación cruzada estratificada y evaluación final sobre un conjunto de prueba no visto.

---

## 🧾 Requisitos

- Python 3.10+
- Scikit-learn
- XGBoost
- LightGBM
- Imbalanced-learn
- Matplotlib
- Pandas
- NumPy

---

## 📚 Créditos Académicos

Este proyecto fue desarrollado con fines educativos en el contexto de un curso de Ciencia de Datos, utilizando información pública del **Ministerio de Salud del Perú (MINSA)**.



