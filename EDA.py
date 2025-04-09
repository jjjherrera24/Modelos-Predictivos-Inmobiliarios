# Importación de librerías
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar el dataset
df = pd.read_csv(r'C:\\Users\\jjjhe\\OneDrive\\Desktop\\Maestría Analítica de Datos\\Modelos Predictivos\\recursos\\raw_sales_original.csv')

# Asegurarse de que la columna 'datesold' sea de tipo datetime
df['datesold'] = pd.to_datetime(df['datesold'], errors='coerce')  # Convierte las fechas, los errores serán convertidos a NaT

# Transformar la columna 'postcode' a categórica
df['postcode'] = df['postcode'].astype('object')



print("\nPrimeras filas del dataset:")
print(df.head())

print("\nInformación general del dataframe:")
df.info()

# Resumen estadístico para variables numéricas
print("\nResumen estadístico de variables numéricas:")
print(df.describe())

# Verificar valores nulos en el dataset
print("\nCantidad de valores nulos por columna:")
print(df.isnull().sum())

# Verificación de la cantidad de valores únicos en variables categóricas
print("\nCantidad de valores únicos en variables categóricas:")
for col in df.select_dtypes(include=['object']).columns:
    print(f"{col}: {df[col].nunique()} valores únicos")

# Análisis de las frecuencias de las variables categóricas
print("\nFrecuencia de las variables categóricas:")
for col in df.select_dtypes(include=['object']).columns:
    print(f"\n{col} - Frecuencia de categorías:")
    print(df[col].value_counts())

# Visualización de la distribución de las variables numéricas
print("\nDistribución de variables numéricas:")
df.select_dtypes(include=['float64', 'int64']).hist(bins=20, figsize=(15, 10))
plt.tight_layout()
plt.show()

# Boxplot para identificar posibles outliers en variables numéricas
print("\nBoxplot de variables numéricas:")
for col in df.select_dtypes(include=['float64', 'int64']).columns:
    plt.figure(figsize=(8, 5))
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot de {col}")
    plt.show()

# Correlación entre variables numéricas
print("\nMatriz de correlación de las variables numéricas:")
dfcorr = df[['datesold', 'price', 'bedrooms']]
correlation_matrix = dfcorr.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Matriz de Correlación')
plt.show()


if 'datesold' in df.columns:
    df.set_index('datesold', inplace=True)
    if 'price' in df.columns:  # Asegúrate de que la columna 'sales' existe en el dataset
        df.resample('Q').sum()['price'].plot(figsize=(12, 6), title="Ventas por Trimestre")
        plt.show()

# Comprobación de valores atípicos utilizando el rango intercuartílico (IQR)
print("\nDetección de outliers usando IQR:")
for col in df.select_dtypes(include=['float64', 'int64']).columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))]
    print(f"{col}: {len(outliers)} valores atípicos detectados.")

# Visualización de las relaciones entre variables numéricas
print("\nVisualización de relaciones entre variables numéricas:")
sns.pairplot(df.select_dtypes(include=['float64', 'int64']))
plt.show()
