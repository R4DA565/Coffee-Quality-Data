# Prueba5
Evaluación Final - Gerardo Garcia
En algunas partes del codigo si es entendible, explicare algunas lineas

categorical_vars = df.select_dtypes(include='object').columns
num_categorical_vars = len(categorical_vars)
print("Número de variables categóricas:", num_categorical_vars)
for var in categorical_vars:
    unique_values = df[var].nunique()
    print(f"Número de valores únicos para {var}: {unique_values}")

el código te ayuda a entender y explorar las variables categóricas en tu conjunto de datos. Puedes descubrir cuántas variables 
categóricas tienes y ver cuántos valores únicos hay en cada una. Esto puede ser útil para comprender la diversidad de categorías en los 
datos y tomar decisiones basadas en esa información.


missing_values = df.isnull().sum()
print("Valores faltantes:")
print(missing_values)

Al ejecutar este código, obtendrás una visión general de los valores faltantes en tu conjunto de datos. Se puede identificar 
fácilmente las columnas que tienen valores faltantes y cuantificar cuántos valores faltantes hay en cada una. Esta información es útil 
para decidir cómo manejar los valores faltantes en tu análisis de datos y realizar acciones como imputación de datos o eliminación de 
filas o columnas con valores faltantes, según sea necesario.


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv("df_arabica_clean.csv")
numeric_data = df.select_dtypes(include=[np.number]).values
column_labels = list(df.select_dtypes(include=[np.number]).columns)
fig, ax = plt.subplots(figsize=(10, 6))
ax.boxplot(numeric_data, vert=False)
ax.set_yticklabels(column_labels)
ax.set_xlabel("Valor")
ax.set_ylabel("Atributos")
ax.set_title("Estadísticas resumidas con Cuartiles")
plt.show()

Al ejecutar este código, se cargará el archivo CSV en un DataFrame, se seleccionarán las 
columnas numéricas, se generará un diagrama de caja para visualizar las estadísticas resumidas de los datos y 
se mostrará el gráfico en pantalla. Esto te ayudará a comprender la distribución, los valores atípicos y los cuartiles 
de las variables numéricas en los datos.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Cargar el archivo df_arabica_clean.csv
df = pd.read_csv("df_arabica_clean.csv")

# Eliminar las columnas no numéricas o convertirlas en numéricas
df_numeric = df.select_dtypes(include=[float, int])

# Separar los datos en características (X) y variable objetivo (y)
X = df_numeric.drop("Overall", axis=1)
y_actual = df_numeric["Overall"]

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y_actual, test_size=0.2, random_state=42)

# Entrenar tu modelo
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# Hacer las predicciones en los datos de prueba
y_pred = modelo.predict(X_test)

# Calcular el MSE y el MAE
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("Mean Squared Error (MSE):", mse)
print("Mean Absolute Error (MAE):", mae)

este código carga un archivo CSV, prepara los datos numéricos, entrena un modelo de regresión lineal, 
realiza predicciones en los datos de prueba y calcula y muestra las métricas de evaluación (MSE y MAE) 
para evaluar el rendimiento del modelo de regresión lineal.

