# -*- coding: utf-8 -*-
"""PROYACTO_MINERIADEDATOS_CLASIFICACION__v1.ipynb

# APLICACIÓN DE LA METODOLOGÍA CRISP-DM PARA PREDECIR LAS ATENCIONES MÉDICAS EN EL HOSPITAL DE APOYO MARIA AUXILIADORA

## IMPORTAR lIBRERIAS
"""

# Librerías base
import pandas as pd
import numpy as np
import warnings

# Visualización
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocesamiento y escalado
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import make_pipeline

# Codificación
import category_encoders as ce

# Desbalanceo
from imblearn.combine import SMOTETomek

# Modelado
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# Entrenamiento y evaluación
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# Opcional
warnings.filterwarnings("ignore")

"""## CARGAR NUESTROS DATOS"""

# Leer datos
data = pd.read_csv("DATOS HOSPITAL MARIA AUXILIADORA.csv",sep=";")
data.tail()

"""## CONOCIENDO NUESTROS DATOS

### 1) Verificación de la calidad de los datos

#### Informacion de las variables
"""

# Informacion de las variables
data.info()

print(data["SEXO"].unique())
print(data["SEGURO"].unique())
print(data["PRESENCIAL_REMOTO"].unique())
print(data['MONTO'].unique())
print(data["ATENDIDO"].unique())

"""####  Cantidad he informacion de datos faltantes"""

# Datos faltantes
data.isnull().sum()

v_numericas = data[['EDAD','MONTO']]
v_numericas.describe()

# valores negativos de la columa EDAD
edad_negative = data[data['EDAD'] < 0]
edad_negative

v_numericas = list(v_numericas.columns)
v_numericas

# Criando subplots para cada columna
plt.figure(figsize=(7, 5))
for i, columna in enumerate(v_numericas):
    plt.subplot(1, len(v_numericas), i+1)
    sns.boxplot(data=data, y=columna)
    plt.title(f"Boxplot de {columna}")
plt.tight_layout()
plt.show()

"""### 2) Exploración de los datos

####  Descripcion y graficos de variables cualitativos y cuantitativos
"""

data.columns

order = data['ESPECIALIDAD'].value_counts().index
plt.figure(figsize=(20, 8))
sns.countplot(data=data, x="ESPECIALIDAD", order = order, palette= 'blend:#7AB,#EDA')
#annotate_bars_percentage(ax)
plt.title("CONTEO DE ESPECIALIDAD")
plt.xticks(rotation=90)
plt.show()

data['ESPECIALIDAD'].value_counts(ascending=True).head(20)

order = data['ESPECIALIDAD'].value_counts().index
plt.figure(figsize=(20, 8))
sns.countplot(data=data, x="ESPECIALIDAD", hue = 'ATENDIDO', order = order, palette= 'blend:#7AB,#EDA')
#annotate_bars_percentage(ax)
plt.title("CONTEO DE ESPECIALIDAD POR ATENCION")
plt.xticks(rotation=90)
plt.show()

# Lista de columnas a graficar
columns_to_plot = ['SEXO', 'SEGURO', 'PRESENCIAL_REMOTO','MONTO']

# Crear gráficos para cada columna
for column in columns_to_plot:
    order = data[column].value_counts().index
    plt.figure(figsize=(7, 4))
    ax = sns.countplot(data=data, x=column, order=order,hue ="ATENDIDO" ,palette="blend:#7AB,#EDA")
    for container in ax.containers:
        ax.bar_label(container, fmt='%d', fontweight='normal', fontsize=10)
    #annotate_bars_percentage(ax
    plt.title(f"FRECUENCIA DE {column} POR ATENCION")
    #plt.xticks(rotation=90)
    plt.show()

order = data['ATENDIDO'].value_counts().index
# Crear gráficos de pastel para cada columna
plt.figure(figsize=(7, 7))
data['ATENDIDO'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, colors=sns.color_palette('pastel'))
plt.title(f"Distribución de {column}")
plt.ylabel('') # Eliminar la etiqueta del eje y por defecto en los gráficos de pastel
plt.show()

plt.figure(figsize=(7, 4))
sns.histplot(data, x="EDAD", bins=20)
plt.title("HISTOGRAMA DE EDAD")
plt.xlabel("EDAD")
plt.ylabel("FRECUENCIA")
plt.show()

# Lista de columnas a graficar
columns_to_plot = ['ATENDIDO','SEXO', 'SEGURO', 'PRESENCIAL_REMOTO', 'MONTO']

# Crear gráficos para cada columna
for column in columns_to_plot:
    g = sns.FacetGrid(data, col=column,height=3.2, aspect=1.2)
    g.map(sns.histplot, "EDAD",bins=20)

# Lista de columnas a graficar
columns_to_plot = ['SEXO', 'SEGURO', 'PRESENCIAL_REMOTO', 'MONTO']

# Crear gráficos para cada columna
for column in columns_to_plot:
    plt.figure(figsize=(9, 5))
    sns.catplot(data=data, x="ATENDIDO", y="EDAD",hue=column, kind="box")
    plt.title(f"DIAGRAMA DE CAJAS DE EDAD POR {column}")
    plt.xlabel("EDAD")
    plt.ylabel("FRECUENCIA")
    plt.show()

# Lista de columnas a graficar
columns_to_plot = ['SEXO', 'SEGURO', 'PRESENCIAL_REMOTO', 'MONTO']

# Crear gráficos para cada columna
for column in columns_to_plot:
    plt.figure(figsize=(9, 5))
    sns.catplot(data=data, x="ATENDIDO", y="EDAD", hue=column, kind="violin")
    plt.title(f"DIAGRAMA DE CAJS DE EDAD POR {column}")
    plt.xlabel("EDAD")
    plt.ylabel("FRECUENCIA")
    plt.show()

"""#### correlacion de variables"""

# Select numeric and object columns
v_numericas4 = data.select_dtypes(include=['float64', 'int64', 'object'])
v_numericas4['ATENDIDO'] = v_numericas4['ATENDIDO'].map({'SI': 1, 'NO': 0})
# Factorize object columns within v_numericas4
for col in v_numericas4.select_dtypes(include=['object']).columns:
    v_numericas4[col] = pd.factorize(v_numericas4[col])[0]
v_numericas4 = v_numericas4.drop(['FECHA_CORTE','ID','DIA_SOLICITACITA','DIA_CITA','DEPARTAMENTO','PROVINCIA','DISTRITO','UBIGEO'],axis=1)
correlaciones1 = v_numericas4.corr()
f, ax = plt.subplots(figsize=(7,7))
sns.heatmap(correlaciones1,cmap="coolwarm",vmin=-1,vmax=1,
            linewidths=.5,square=True,annot=True)

"""## PREPARACIÓN DE LOS DATOS

### 1) Selección de los datos
"""

data.columns

data1 = data.iloc[:, 2:11]
data1 = data1.drop(['SEXO'],axis=1)
data1.head()

"""### 2) Limpieza de los datos

#### Corregir los tipos de datos
"""

def corregir_tipos_de_datos(data1):

  # Convetir 'SI' y 'NO' a 1 y 0
  if 'ATENDIDO' in data1.columns:
    data1['ATENDIDO']= data1['ATENDIDO'].map({'SI':1, 'NO':0})
  # Convertir columnas de fecha a tipo datatime
  for col in ['DIA_SOLICITACITA', 'DIA_CITA']:
    if col in data.columns:
      data1[col]= pd.to_datetime(data1[col].astype(str), errors="coerce")

  return data1

data1 = corregir_tipos_de_datos(data1)

data1.info()
#data1.columns

"""#### Limpieza de registros"""

def limpiar_datos(data1):
  # filtrar especialidades frecuentes
  data1 = data1[data1['ESPECIALIDAD'].map(data1['ESPECIALIDAD'].value_counts()) >= 365]
  #Eliminar nulos
  data1 = data1.dropna(subset=['EDAD','SEGURO'])
  # Edad a valor absoluto
  data1['EDAD']= data['EDAD'].abs()
  # Reiniciar indice
  return data1.reset_index(drop=True)

data1 = limpiar_datos(data1)

data1.info()

"""### 3) Construccion de datos

#### Crear variables
"""

def crear_variables(data):
    """
    Crea variables derivadas de fechas para análisis.
    Requiere columnas 'DIA_CITA' y 'DIA_SOLICITACITA' en formato datetime.

    - DIFERENCIA_DIAS: días entre solicitud y cita
    - mes_cita: mes de la cita
    - estacion_cita: estación del año (hemisferio sur), convertida a entero
    - semana_mes_cita: semana del mes (1 a 5)
    - bimestre_cita: bimestre del año (1 a 6)
    - trimestre_cita: trimestre del año (1 a 4)
    - Cita_mes_diferente: 1 si el mes de cita es distinto al de solicitud
    - semestre_cita: semestre del año (1 o 2)
    """

    def obtener_estacion(fecha):
        año = fecha.year
        if pd.Timestamp(f'{año}-12-21') <= fecha or fecha < pd.Timestamp(f'{año}-3-21'):
            return 'verano'
        elif pd.Timestamp(f'{año}-3-21') <= fecha < pd.Timestamp(f'{año}-6-21'):
            return 'otoño'
        elif pd.Timestamp(f'{año}-6-21') <= fecha < pd.Timestamp(f'{año}-9-23'):
            return 'invierno'
        else:
            return 'primavera'

    # Calcular diferencia en días
    data['DIFERENCIA_DIAS'] = (data['DIA_CITA'] - data['DIA_SOLICITACITA']).dt.days

    # Extraer componentes de fecha
    data['mes_cita'] = data['DIA_CITA'].dt.month
    data['semana_mes_cita'] = (data['DIA_CITA'].dt.day.sub(1) // 7) + 1
    data['bimestre_cita'] = ((data['DIA_CITA'].dt.month - 1) // 2) + 1
    data['trimestre_cita'] = ((data['DIA_CITA'].dt.month - 1) // 3) + 1
    data['semestre_cita'] = ((data['DIA_CITA'].dt.month - 1) // 6) + 1
    data['Cita_mes_diferente'] = (data['DIA_SOLICITACITA'].dt.month != data['DIA_CITA'].dt.month).astype(int)

    # Estación del año y codificación
    data['estacion_cita'] = data['DIA_CITA'].apply(obtener_estacion)
    data['estacion_cita'] = pd.factorize(data['estacion_cita'])[0]

    return data

data1 = crear_variables(data1)

data1.info()

"""#### * Correlacion con las nuevas variables"""

# Select numeric and object columns
v_numericas5 = data1.iloc[:, [6] + list(range(8,16))]
# Factorize object columns within v_numericas5
for col in v_numericas5.select_dtypes(include=['object']).columns:
    v_numericas5[col] = pd.factorize(v_numericas5[col])[0]
correlaciones2 = v_numericas5.corr()
import matplotlib.pyplot as plt
import seaborn as sns
f, ax = plt.subplots(figsize=(10,10))
sns.heatmap(correlaciones2,cmap="coolwarm",vmin=-1,vmax=1,
            linewidths=.5,square=True,annot=True)

# Select the columns and convert their type to 'category'
cols= data1.columns[[7] + list(range(9,16))]
data1[cols] = data1[cols].astype('category')

# Verificar los tipos de datos actualizados
data1.info()

"""####  Descripcion de la nueva variable columnas"""

data1['DIFERENCIA_DIAS'].describe()

df_diasn = data1[data1['DIFERENCIA_DIAS'] < 0]
df_diasn

"""#### Limpieza de la nueva variable"""

def limpiar_diferencia_dias(data):
    return data[data['DIFERENCIA_DIAS'] >= 0].reset_index(drop=True)

data1 = limpiar_diferencia_dias(data1)

data1.info()

# Seleccionar columnas por posición
df_iloc = data1.iloc[:, 2:9]

# Seleccionar columnas por nombre
df_names = data1[['semestre_cita', 'Cita_mes_diferente', 'semana_mes_cita']]

# Concatenar ambas partes
data2 = pd.concat([df_iloc, df_names], axis=1)

"""### 4) Formateo de los datos

####  Escalamiento de variables numericos
"""

v_numericos = data2.select_dtypes([int,float]).drop(['ATENDIDO'],axis=1)
v_numericos.columns

from sklearn.preprocessing import MinMaxScaler
v_numericos_minmax = pd.DataFrame(MinMaxScaler().fit_transform(v_numericos), columns=v_numericos.columns)

#print("Min-Max Scaling:")
#v_numericos_minmax.head()

'''from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
v_numericos_std = scaler.fit_transform(v_numericos)
v_numericos_std = pd.DataFrame(v_numericos_std, columns=v_numericos.columns)
v_numericos_std.head()'''

"""####  Duminizacion de variables"""

v_dum = data2.select_dtypes(["category","object"])
#v_dum.drop(['mes_cita','estacion_cita', 'bimestre_cita'], axis =1, inplace=True)
v_dum.columns

encoder = ce.OneHotEncoder(cols=v_dum.columns, use_cat_names=True)
# Aplicamos la transformación de OneHotEncoder
v_dum_encoded_df = encoder.fit_transform(v_dum)
#v_dum_encoded_df .head()

data3 = pd.concat([v_numericos_minmax,v_dum_encoded_df],axis=1)
data3.shape

X = data3
y = data2['ATENDIDO']
y.unique()

"""#### Tratamiento de desbalance de clases"""

plt.figure(figsize=(5, 4))
sns.countplot(x='ATENDIDO',data= data2,palette='hls')
plt.show()
print(data2['ATENDIDO'].value_counts())

X_bal, y_bal = SMOTETomek(sampling_strategy="minority", n_jobs=-1).fit_resample(X, y)

plt.figure(figsize=(5,4))
sns.countplot(x=y_bal, palette="hls")
plt.show()
print(y_bal.value_counts())

"""## MODELADO

### 1) Particion 80/20
"""

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_bal, y_bal, test_size=0.2, random_state=42, stratify=y_bal)

print("Dimensiones de los conjuntos de entrenamiento y prueba:")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

"""### 2) Modelo Lightgbm"""

from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score

# Inicializar el modelo LightGBM
model_lgbm = LGBMClassifier(
    random_state=42,
    force_row_wise=True,

    # Boosting y árboles
    boosting_type='gbdt',
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=31,

    # Regularización
    reg_alpha=0.1,
    reg_lambda=1.0,

    # Control de overfitting
    max_depth=-1,
    min_child_samples=20,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,

    # Optimización
    n_jobs=-1,
    verbosity=-1
)

# Realizar validación cruzada con 5 folds, calculando F1
cv_results = cross_val_score(model_lgbm, X_train, y_train, cv=5, scoring='f1')

print("Resultados de la validación cruzada:")

# Print F1 scores
print("\nMétricas F1 por fold:")
print(cv_results)
print(f"F1 promedio: {cv_results.mean():.4f}")
print(f"Desviación estándar F1: {cv_results.std():.4f}")

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# Entrenar el modelo con los datos de entrenamiento
model_lgbm.fit(X_train, y_train)

# Predecir en el conjunto de prueba
y_pred = model_lgbm.predict(X_test)
y_pred_proba = model_lgbm.predict_proba(X_test)[:, 1]  # Para ROC AUC

# Evaluar el modelo
print("\n Reporte de Clasificación:")
print(classification_report(y_test, y_pred, digits=4))

print("Matriz de Confusión:")
print(confusion_matrix(y_test, y_pred))

# Métricas adicionales individuales
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print("\n Métricas Individuales:")
print(f"Accuracy       : {accuracy:.4f}")
print(f"Precision      : {precision:.4f}")
print(f"Recall         : {recall:.4f}")
print(f"F1 Score       : {f1:.4f}")
print(f"ROC AUC Score  : {roc_auc:.4f}")

"""### 3) Model Xgboost"""

from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score

# Inicializar el modelo XGBoost
model_xgb = XGBClassifier(
    random_state=42,
    n_estimators=1000,           # número de árboles
    learning_rate=0.05,          # tasa de aprendizaje
    max_depth=6,                 # profundidad máxima (num_leaves equivalente ≈ 2^max_depth)
    subsample=0.8,               # porcentaje de filas para cada árbol (row sampling)
    colsample_bytree=0.8,        # porcentaje de columnas para cada árbol (feature sampling)

    # Regularización
    reg_alpha=0.1,               # L1
    reg_lambda=1.0,              # L2

    n_jobs=-1,                   # usar todos los núcleos disponibles
    verbosity=0,                 # silenciar salidas de consola
    use_label_encoder=False,     # evitar warning de codificador interno
    device='gpu'                 # usar GPU
)

# Realizar validación cruzada con 5 folds, calculando F1
cv_results = cross_val_score(model_xgb, X_train, y_train, cv=5, scoring='f1')

print("Resultados de la validación cruzada con XGBoost:")

# Print F1 scores
print("\nMétricas F1 por fold:")
print(cv_results)
print(f"F1 promedio: {cv_results.mean():.4f}")
print(f"Desviación estándar F1: {cv_results.std():.4f}")

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# Entrenar el modelo con los datos de entrenamiento
model_xgb.fit(X_train, y_train)

# Predecir en el conjunto de prueba
y_pred = model_xgb.predict(X_test)
y_pred_proba = model_xgb.predict_proba(X_test)[:, 1]  # Para ROC AUC

# Evaluar el modelo
print("\n Reporte de Clasificación:")
print(classification_report(y_test, y_pred, digits=4))

print("Matriz de Confusión:")
print(confusion_matrix(y_test, y_pred))

# Métricas adicionales individuales
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print("\n Métricas Individuales:")
print(f"Accuracy       : {accuracy:.4f}")
print(f"Precision      : {precision:.4f}")
print(f"Recall         : {recall:.4f}")
print(f"F1 Score       : {f1:.4f}")
print(f"ROC AUC Score  : {roc_auc:.4f}")

"""### 3) Model SVM"""

from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# SVM requiere escalado previo → usamos un pipeline para encadenar escalado + modelo
model_svm = make_pipeline(
    StandardScaler(),  # escalado de características
    SVC(
        kernel='rbf',
        C=1.0,
        gamma='scale',
        probability=True,
        random_state=42
    )
)

# Validación cruzada con F1
cv_results = cross_val_score(model_svm, X_train, y_train, cv=5, scoring='f1', n_jobs=-1)

print("Resultados de la validación cruzada con SVM:")

# Print F1 scores
print("\nMétricas F1 por fold:")
print(cv_results)
print(f"F1 promedio: {cv_results.mean():.4f}")
print(f"Desviación estándar F1: {cv_results.std():.4f}")

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# Entrenar el modelo con los datos de entrenamiento
model_svm.fit(X_train, y_train)

# Predecir en el conjunto de prueba
y_pred = model_svm.predict(X_test)
y_pred_proba = model_svm.predict_proba(X_test)[:, 1]  # Para ROC AUC

# Evaluar el modelo
print("\n Reporte de Clasificación:")
print(classification_report(y_test, y_pred, digits=4))

print("Matriz de Confusión:")
print(confusion_matrix(y_test, y_pred))

# Métricas adicionales individuales
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print("\n Métricas Individuales:")
print(f"Accuracy       : {accuracy:.4f}")
print(f"Precision      : {precision:.4f}")
print(f"Recall         : {recall:.4f}")
print(f"F1 Score       : {f1:.4f}")
print(f"ROC AUC Score  : {roc_auc:.4f}")

"""### 3) Model Random Forest"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# Inicializar el modelo Random Forest
model_rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    bootstrap=True,
    random_state=42,
    n_jobs=-1,
    verbose=0
)

# Validación cruzada con F1, usando todos los CPUs
cv_results = cross_val_score(model_rf, X_train, y_train, cv=5, scoring='f1', n_jobs=-1)

print("Resultados de la validación cruzada con Random Forest:")

# Print F1 scores
print("\nMétricas F1 por fold:")
print(cv_results)
print(f"F1 promedio: {cv_results.mean():.4f}")
print(f"Desviación estándar F1: {cv_results.std():.4f}")

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# Entrenar el modelo con los datos de entrenamiento
model_rf.fit(X_train, y_train)

# Predecir en el conjunto de prueba
y_pred = model_rf.predict(X_test)
y_pred_proba = model_rf.predict_proba(X_test)[:, 1]  # Para ROC AUC

# Evaluar el modelo
print("\n Reporte de Clasificación:")
print(classification_report(y_test, y_pred, digits=4))

print("Matriz de Confusión:")
print(confusion_matrix(y_test, y_pred))

# Métricas adicionales individuales
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print("\n Métricas Individuales:")
print(f"Accuracy       : {accuracy:.4f}")
print(f"Precision      : {precision:.4f}")
print(f"Recall         : {recall:.4f}")
print(f"F1 Score       : {f1:.4f}")
print(f"ROC AUC Score  : {roc_auc:.4f}")



