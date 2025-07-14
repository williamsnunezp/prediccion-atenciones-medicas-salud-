# Predicción de Atenciones Médicas en el Hospital María Auxiliadora

Este proyecto académico implementa un flujo de **Machine Learning** para predecir si un paciente será **atendido o no** en el Hospital María Auxiliadora, Lima – Perú. Se emplea la metodología **CRISP-DM**, abordando en codigo específicamente las fases **2 a 5**:  
**Comprensión de los datos, Preparación de los datos, Modelado y Evaluación**.

🔗 **Fuente de datos**: [Portal de Datos Abiertos del Gobierno del Perú](https://datosabiertos.gob.pe/group/hospital-mar%C3%ADa-auxiliadora?sort_by=changed&f%5B0%5D=changed%3A2024-05-02)

---

## 🎯 Objetivo del proyecto

Desarrollar un modelo predictivo que, a partir de variables administrativas y derivados de tiempo de las citas médicas, **anticipe la probabilidad de que un paciente sea atendido** o no. Esto puede ayudar a mejorar la gestión hospitalaria y reducir pérdidas de recursos por inasistencias.

---

## 🧭 Metodología CRISP-DM aplicada

| Fase                |   Descripción                                                                 |
|---------------------|---------------------------------------------------------------------------------------|
| Fase 1: Negocio     | ✅ Definición del objetivo del proyecto                                   |
| Fase 2: Datos       | ✅ Carga del dataset, exploración visual, análisis de calidad, valores nulos  |
| Fase 3: Preparación | ✅ Limpieza, transformación, ingeniería de variables temporales y categóricas |
| Fase 4: Modelado    | ✅ Entrenamiento con LightGBM, XGBoost, SVM y Random Forest                    |
| Fase 5: Evaluación  | ✅ Validación cruzada, métricas (Accuracy, F1, AUC) y análisis de desempeño    |
| Fase 6: Despliegue  | ✅ Desarrollado como propuesta                                                                        |

---

## 🛠️ Algoritmos utilizados

- LightGBM
- XGBoost
- Random Forest
- Support Vector Machines (SVM)

Se usó balanceo de clases con **SMOTETomek** y escalado de variables.

---

## 📊 Variables destacadas

- Edad
- Tipo de seguro
- Modalidad de cita (presencial/remoto)
- Especialidad médica
- Diferencia de días entre solicitud y cita
- Estación del año, bimestre, semana del mes, entre otros derivados de fechas

---

## 📈 Resultados preliminares

| Modelo        | F1 Score Promedio (CV) |
|---------------|------------------------|
| LightGBM      | ~0.83                  |
| XGBoost       | ~0.79                  |
| Random Forest | ~0.80                  |
| SVM           | ~0.78                  |

*La evaluación se basó en validación cruzada estratificada con 5 folds.*

---

## 📚 Créditos académicos
Este trabajo fue desarrollado con fines formativos en Ciencia de Datos, a partir de información pública del Ministerio de Salud del Perú (MINSA) y su portal de datos abiertos.


