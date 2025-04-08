# Importar las bibliotecas necesarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

# Función para realizar un EDA rápido
def quick_eda(df):
    # 1. Ver las primeras filas del dataset
    print("Primeras filas del dataset:")
    df.head()
    
    # 2. Información general sobre el dataset (tipos de datos y valores nulos)
    print("\nInformación general:")
    df.info()
    
    # 3. Estadísticas descriptivas de las columnas numéricas
    print("\nEstadísticas descriptivas:")
    print(df.describe())
    
    # 4. Ver la cantidad de valores nulos
    print("\nCantidad de valores nulos por columna:")
    print(df.isnull().sum())
    
    # 5. Visualizar la cantidad de valores nulos
    print("\nVisualización de valores nulos:")
    msno.matrix(df)
    plt.show()
    
    # 6. Visualización de la distribución de variables numéricas
    print("\nDistribución de variables numéricas:")
    df.select_dtypes(include=[np.number]).hist(figsize=(10, 8), bins=20)
    plt.tight_layout()
    plt.show()
    
    # 7. Ver la correlación entre variables numéricas
    print("\nMapa de calor de correlaciones:")
    corr = df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.show()
    
    # 8. Visualización de distribuciones de variables categóricas
    print("\nDistribución de variables categóricas:")
    for col in df.select_dtypes(include=[object]).columns:
        plt.figure(figsize=(8, 6))
        sns.countplot(data=df, x=col)
        plt.title(f'Distribución de la columna {col}')
        plt.xticks(rotation=45)
        plt.show()

# Ejemplo de uso:
# Cargar el dataset
df = pd.read_csv(r'C:\\Users\\jjjhe\\OneDrive\\Desktop\\Maestría Analítica de Datos\\Modelos Predictivos\\recursos\\raw_sales.csv')

# Llamar a la función de EDA rápido
quick_eda(df)