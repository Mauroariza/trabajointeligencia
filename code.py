# Importamos las bibliotecas necesarias para el proyecto
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
import re
import warnings
warnings.filterwarnings('ignore')

# Configuramos el estilo de las visualizaciones
plt.style.use('fivethirtyeight')
sns.set_palette("Set2")

print("Paso 1: Definición del problema completado")

# Cargamos el dataset
ruta_archivo = 'Registro_de_Activos_de_Informaci_n_20250327.csv'
datos = pd.read_csv(ruta_archivo, encoding='latin1')

# Exploramos las primeras filas para entender la estructura
print("\nPrimeros registros del dataset:")
print(datos.head())

# Información general del dataset
print("\nInformación del dataset:")
print(datos.info())

# Estadísticas descriptivas
print("\nEstadísticas descriptivas:")
print(datos.describe(include='all'))

# Verificamos los valores únicos en la columna objetivo
print("\nDistribución de la variable objetivo (Categoria):")
print(datos['Categoria'].value_counts())

# Verificamos valores nulos
print("\nValores nulos por columna:")
print(datos.isnull().sum())

# Obtenemos los nombres reales de las columnas para verificar
print("\nNombres exactos de las columnas:")
for col in datos.columns:
    print(f"- '{col}'")

print("\nPaso 2: Identificación y carga de datos completado")

# Hacemos una copia de los datos para no modificar los originales
datos_preprocesados = datos.copy()

# CORRECCIÓN: Obtenemos el nombre exacto de la columna de información personal
columna_info_personal = [col for col in datos.columns if "Contiene informacion personal" in col][0]

# Seleccionamos las columnas relevantes para nuestro análisis
columnas_seleccionadas = ['Proceso', 'Tipo de Activo', 'Categoria', 'Idioma', 'Origen', 
                         'Estado', columna_info_personal]
datos_filtrados = datos_preprocesados[columnas_seleccionadas]

# Reemplazamos los valores "No Aplica" por NaN para facilitar su manejo
datos_filtrados.replace('No Aplica', np.nan, inplace=True)

# Verificamos la proporción de valores nulos después de la transformación
print("\nProporción de valores nulos después de la transformación:")
print(datos_filtrados.isnull().mean())

# Creamos una función para extraer información útil de la descripción del activo
def extraer_longitud_descripcion(row):
    if pd.isna(row):
        return 0
    return len(str(row))

# Aplicamos la función para crear una nueva característica
datos_filtrados['Longitud_Descripcion'] = datos['Descripcion'].apply(extraer_longitud_descripcion)

# Codificamos variables categóricas
le_proceso = LabelEncoder()
le_tipo = LabelEncoder()
le_idioma = LabelEncoder()
le_origen = LabelEncoder()
le_estado = LabelEncoder()
le_info_personal = LabelEncoder()

# Aplicamos la codificación manejando valores nulos
datos_filtrados['Proceso_encoded'] = le_proceso.fit_transform(datos_filtrados['Proceso'].fillna('Desconocido'))
datos_filtrados['Tipo_encoded'] = le_tipo.fit_transform(datos_filtrados['Tipo de Activo'].fillna('Desconocido'))
datos_filtrados['Idioma_encoded'] = le_idioma.fit_transform(datos_filtrados['Idioma'].fillna('Desconocido'))
datos_filtrados['Origen_encoded'] = le_origen.fit_transform(datos_filtrados['Origen'].fillna('Desconocido'))
datos_filtrados['Estado_encoded'] = le_estado.fit_transform(datos_filtrados['Estado'].fillna('Desconocido'))
datos_filtrados['Info_Personal_encoded'] = le_info_personal.fit_transform(datos_filtrados[columna_info_personal].fillna('Desconocido'))

# Preparamos los datos para los modelos
X = datos_filtrados[['Proceso_encoded', 'Tipo_encoded', 'Idioma_encoded', 
                    'Origen_encoded', 'Estado_encoded', 'Info_Personal_encoded',
                    'Longitud_Descripcion']]
y = datos_filtrados['Categoria'].fillna('Desconocido')

# Codificamos la variable objetivo
le_categoria = LabelEncoder()
y_encoded = le_categoria.fit_transform(y)

print("\nPrimeras filas después del preprocesamiento:")
print(datos_filtrados.head())

# Visualizamos la distribución de la variable objetivo
plt.figure(figsize=(10, 6))
sns.countplot(x='Categoria', data=datos_filtrados)
plt.title('Distribución de Categorías de Activos')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('distribucion_categorias.png')
plt.show()

# Exploramos relaciones entre variables
plt.figure(figsize=(12, 8))
sns.heatmap(datos_filtrados[['Proceso_encoded', 'Tipo_encoded', 'Idioma_encoded', 
                            'Origen_encoded', 'Estado_encoded', 'Info_Personal_encoded',
                            'Longitud_Descripcion']].corr(), annot=True, cmap='coolwarm')
plt.title('Matriz de Correlación entre Variables')
plt.tight_layout()
plt.savefig('matriz_correlacion.png')
plt.show()

# Analizamos la relación entre Tipo de Activo y Categoría
plt.figure(figsize=(12, 6))
sns.countplot(x='Tipo de Activo', hue='Categoria', data=datos_filtrados)
plt.title('Distribución de Categorías por Tipo de Activo')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('categorias_por_tipo.png')
plt.show()

print("\nPaso 3: Preparación y preprocesamiento de datos completado")

# Dividimos los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)

print(f"\nTamaño del conjunto de entrenamiento: {X_train.shape[0]} muestras")
print(f"Tamaño del conjunto de prueba: {X_test.shape[0]} muestras")

# Normalizamos las características numéricas
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 1. Modelo de Máquina de Soporte Vectorial (SVM)
print("\n=== Entrenando Máquina de Soporte Vectorial ===")
modelo_svm = SVC(kernel='rbf', C=10, gamma=0.1, random_state=42)
modelo_svm.fit(X_train_scaled, y_train)

# Evaluamos el modelo SVM
y_pred_svm = modelo_svm.predict(X_test_scaled)
precision_svm = accuracy_score(y_test, y_pred_svm)
print(f"Precisión del modelo SVM: {precision_svm:.4f}")
print("\nInforme de clasificación SVM:")
print(classification_report(y_test, y_pred_svm))

# Matriz de confusión para SVM
plt.figure(figsize=(8, 6))
cm_svm = confusion_matrix(y_test, y_pred_svm)
sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=le_categoria.classes_, 
            yticklabels=le_categoria.classes_)
plt.title('Matriz de Confusión - SVM')
plt.ylabel('Etiqueta Real')
plt.xlabel('Etiqueta Predicha')
plt.tight_layout()
plt.savefig('matriz_confusion_svm.png')
plt.show()

# 2. Modelo de Random Forest
print("\n=== Entrenando Random Forest ===")
modelo_rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
modelo_rf.fit(X_train_scaled, y_train)

# Evaluamos el modelo Random Forest
y_pred_rf = modelo_rf.predict(X_test_scaled)
precision_rf = accuracy_score(y_test, y_pred_rf)
print(f"Precisión del modelo Random Forest: {precision_rf:.4f}")
print("\nInforme de clasificación Random Forest:")
print(classification_report(y_test, y_pred_rf))

# Matriz de confusión para Random Forest
plt.figure(figsize=(8, 6))
cm_rf = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', 
            xticklabels=le_categoria.classes_, 
            yticklabels=le_categoria.classes_)
plt.title('Matriz de Confusión - Random Forest')
plt.ylabel('Etiqueta Real')
plt.xlabel('Etiqueta Predicha')
plt.tight_layout()
plt.savefig('matriz_confusion_rf.png')
plt.show()

# 3. Modelo de Regresión Logística
print("\n=== Entrenando Regresión Logística ===")
modelo_lr = LogisticRegression(max_iter=1000, random_state=42)
modelo_lr.fit(X_train_scaled, y_train)

# Evaluamos el modelo de Regresión Logística
y_pred_lr = modelo_lr.predict(X_test_scaled)
precision_lr = accuracy_score(y_test, y_pred_lr)
print(f"Precisión del modelo de Regresión Logística: {precision_lr:.4f}")
print("\nInforme de clasificación Regresión Logística:")
print(classification_report(y_test, y_pred_lr))

# Matriz de confusión para Regresión Logística
plt.figure(figsize=(8, 6))
cm_lr = confusion_matrix(y_test, y_pred_lr)
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', 
            xticklabels=le_categoria.classes_, 
            yticklabels=le_categoria.classes_)
plt.title('Matriz de Confusión - Regresión Logística')
plt.ylabel('Etiqueta Real')
plt.xlabel('Etiqueta Predicha')
plt.tight_layout()
plt.savefig('matriz_confusion_lr.png')
plt.show()

# Comparamos los modelos
modelos = ['SVM', 'Random Forest', 'Regresión Logística']
precisiones = [precision_svm, precision_rf, precision_lr]

plt.figure(figsize=(10, 6))
sns.barplot(x=modelos, y=precisiones)
plt.title('Comparación de Precisión entre Modelos')
plt.ylim(0, 1)
plt.ylabel('Precisión')
plt.tight_layout()
plt.savefig('comparacion_modelos.png')
plt.show()

# Importancia de características (Random Forest)
if hasattr(modelo_rf, 'feature_importances_'):
    importancias = modelo_rf.feature_importances_
    nombres_features = ['Proceso', 'Tipo de Activo', 'Idioma', 'Origen', 
                        'Estado', 'Info Personal', 'Long. Descripción']
    
    indices_ordenados = np.argsort(importancias)[::-1]
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importancias[indices_ordenados], y=[nombres_features[i] for i in indices_ordenados])
    plt.title('Importancia de Características - Random Forest')
    plt.xlabel('Importancia Relativa')
    plt.tight_layout()
    plt.savefig('importancia_caracteristicas.png')
    plt.show()

print("\nPaso 4: Modelado de datos completado")

# Aplicamos K-Means para identificar clusters naturales en los datos
print("\n=== Aplicando K-Means Clustering ===")

# Determinamos el número óptimo de clusters con el método del codo
inertias = []
k_range = range(1, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_train_scaled)
    inertias.append(kmeans.inertia_)

# Visualizamos el método del codo
plt.figure(figsize=(10, 6))
plt.plot(k_range, inertias, 'o-')
plt.title('Método del Codo para K-Means')
plt.xlabel('Número de Clusters (k)')
plt.ylabel('Inercia')
plt.grid(True)
plt.tight_layout()
plt.savefig('metodo_codo.png')
plt.show()

# Seleccionamos un número óptimo de clusters (supongamos k=3 basado en el gráfico)
k_optimo = 3
kmeans = KMeans(n_clusters=k_optimo, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_train_scaled)

# Añadimos las etiquetas de cluster al dataframe para análisis
X_train_df = pd.DataFrame(X_train, columns=['Proceso', 'Tipo', 'Idioma', 'Origen', 
                                           'Estado', 'Info_Personal', 'Long_Descripcion'])
X_train_df['Cluster'] = cluster_labels
X_train_df['Categoria_Real'] = y_train

# Analizamos la distribución de categorías por cluster
plt.figure(figsize=(12, 6))
for i in range(k_optimo):
    cluster_data = X_train_df[X_train_df['Cluster'] == i]['Categoria_Real'].value_counts(normalize=True)
    plt.bar(i, cluster_data.get(0, 0), color='skyblue', label='Primario' if i == 0 else "")
    plt.bar(i, cluster_data.get(1, 0), bottom=cluster_data.get(0, 0), color='salmon', 
           label='De soporte' if i == 0 else "")

plt.title('Distribución de Categorías por Cluster')
plt.xlabel('Cluster')
plt.ylabel('Proporción')
plt.xticks(range(k_optimo))
plt.legend()
plt.tight_layout()
plt.savefig('categorias_por_cluster.png')
plt.show()

# Visualizamos los clusters en 2D (usando las dos características más importantes)
if hasattr(modelo_rf, 'feature_importances_'):
    top_features = np.argsort(modelo_rf.feature_importances_)[-2:]
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_train_scaled[:, top_features[0]], 
                         X_train_scaled[:, top_features[1]], 
                         c=cluster_labels, alpha=0.6, cmap='viridis')
    plt.colorbar(scatter, label='Cluster')
    plt.xlabel(f'Característica {nombres_features[top_features[0]]}')
    plt.ylabel(f'Característica {nombres_features[top_features[1]]}')
    plt.title('Visualización de Clusters en 2D')
    plt.tight_layout()
    plt.savefig('clusters_2d.png')
    plt.show()

# Analizamos las características de cada cluster
print("\nCaracterísticas promedio por cluster:")
cluster_means = X_train_df.groupby('Cluster').mean()
print(cluster_means)

# Comparamos los resultados de clustering con las categorías reales
print("\nDistribución de categorías por cluster:")
for i in range(k_optimo):
    cluster_dist = X_train_df[X_train_df['Cluster'] == i]['Categoria_Real'].value_counts(normalize=True)
    print(f"Cluster {i}:")
    for cat, prop in cluster_dist.items():
        cat_name = le_categoria.inverse_transform([int(cat)])[0]
        print(f"  {cat_name}: {prop:.2%}")

print("\nPaso 5: Aplicación de Clustering completado")

print("\n=== ANÁLISIS DE RESULTADOS Y CONCLUSIONES ===")

# Resumen de rendimiento de modelos
print("\nResumen de rendimiento de modelos:")
modelos_dict = {
    'SVM': precision_svm,
    'Random Forest': precision_rf,
    'Regresión Logística': precision_lr
}

mejor_modelo = max(modelos_dict.items(), key=lambda x: x[1])
print(f"El mejor modelo es {mejor_modelo[0]} con una precisión de {mejor_modelo[1]:.2%}")

# Análisis de errores para el mejor modelo
if mejor_modelo[0] == 'SVM':
    y_pred_mejor = y_pred_svm
elif mejor_modelo[0] == 'Random Forest':
    y_pred_mejor = y_pred_rf
else:
    y_pred_mejor = y_pred_lr

# Creamos un DataFrame para analizar los errores
X_test_df = pd.DataFrame(X_test, columns=['Proceso', 'Tipo', 'Idioma', 'Origen', 
                                          'Estado', 'Info_Personal', 'Long_Descripcion'])
X_test_df['Categoria_Real'] = y_test
X_test_df['Categoria_Predicha'] = y_pred_mejor
X_test_df['Error'] = y_test != y_pred_mejor

# Análisis por tipo de activo
print("\nTasa de error por tipo de activo:")
tipos_unicos = le_tipo.classes_
for i, tipo in enumerate(tipos_unicos):
    tipo_subset = X_test_df[X_test_df['Tipo'] == i]
    if len(tipo_subset) > 0:
        tasa_error = tipo_subset['Error'].mean()
        print(f"{tipo}: {tasa_error:.2%} (n={len(tipo_subset)})")

# Análisis por proceso
print("\nTasa de error por proceso (top 5 con más registros):")
procesos_error = X_test_df.groupby('Proceso')['Error'].agg(['mean', 'count'])
procesos_error = procesos_error[procesos_error['count'] > 5].sort_values('count', ascending=False).head()
print(procesos_error)

# Conclusiones sobre importancia de características
if hasattr(modelo_rf, 'feature_importances_'):
    print("\nImportancia de características para la clasificación:")
    for i in np.argsort(modelo_rf.feature_importances_)[::-1]:
        print(f"{nombres_features[i]}: {modelo_rf.feature_importances_[i]:.4f}")

print("\nPaso 6: Análisis de resultados completado")

# Seleccionamos el mejor modelo para la implementación final
print("\n=== IMPLEMENTACIÓN DE MODELO FINAL ===")

# Basado en los resultados anteriores, seleccionamos el mejor modelo
if mejor_modelo[0] == 'Random Forest':
    modelo_final = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
elif mejor_modelo[0] == 'SVM':
    modelo_final = SVC(kernel='rbf', C=10, gamma=0.1, random_state=42)
else:
    modelo_final = LogisticRegression(max_iter=1000, random_state=42)

# Entrenamos el modelo final con todos los datos
print(f"Entrenando modelo final ({mejor_modelo[0]}) con todos los datos...")
modelo_final.fit(scaler.fit_transform(X), y_encoded)

# Preparamos algunos ejemplos para demostrar la predicción
print("\nEjemplos de predicciones:")

# Creamos algunos ejemplos manualmente
ejemplos = [
    # Un activo típicamente de soporte (como un hardware de computador)
    [0, 0, 0, 0, 0, 0, 100],  
    # Un activo típicamente primario (como un sistema de información)
    [1, 2, 0, 0, 0, 1, 200],   
    # Un caso ambiguo
    [2, 1, 0, 0, 0, 0, 150]    
]

ejemplos_scaled = scaler.transform(ejemplos)
predicciones = modelo_final.predict(ejemplos_scaled)

for i, ejemplo in enumerate(ejemplos):
    categoria_pred = le_categoria.inverse_transform([predicciones[i]])[0]
    print(f"Ejemplo {i+1}: {categoria_pred}")

# Función auxiliar para hacer predicciones con nuevos datos
def predecir_categoria(proceso, tipo_activo, idioma, origen, estado, info_personal, long_descripcion):
    # Codificar entradas categóricas
    proceso_cod = le_proceso.transform([proceso])[0] if proceso in le_proceso.classes_ else -1
    tipo_cod = le_tipo.transform([tipo_activo])[0] if tipo_activo in le_tipo.classes_ else -1
    idioma_cod = le_idioma.transform([idioma])[0] if idioma in le_idioma.classes_ else -1
    origen_cod = le_origen.transform([origen])[0] if origen in le_origen.classes_ else -1
    estado_cod = le_estado.transform([estado])[0] if estado in le_estado.classes_ else -1
    info_cod = le_info_personal.transform([info_personal])[0] if info_personal in le_info_personal.classes_ else -1
    
    # Crear el vector de características
    X_nuevo = np.array([[proceso_cod, tipo_cod, idioma_cod, origen_cod, estado_cod, info_cod, long_descripcion]])
    
    # Escalar y predecir
    X_nuevo_scaled = scaler.transform(X_nuevo)
    prediccion = modelo_final.predict(X_nuevo_scaled)[0]
    
    return le_categoria.inverse_transform([prediccion])[0]

print("\nEjemplo de uso de la función para predecir nuevos activos:")
nueva_prediccion = predecir_categoria(
    proceso="Secretaría de Hacienda y Gestion Financiera",
    tipo_activo="Hardware",
    idioma="Español",
    origen="Interno",
    estado="Activo",
    info_personal="No Aplica",
    long_descripcion=120
)
print(f"Categoría predicha para el nuevo activo: {nueva_prediccion}")

# Mensaje final
print("\n¡Proyecto completado con éxito!")
print("El modelo de clasificación desarrollado puede ser utilizado para automatizar la categorización de activos de información.")
print("\nPaso 7: Implementación del modelo final completado")
