•	raw_sales_original.csv: Archivo original descargado desde Kaggle. Se le cambió el nombre para evitar modificarlo. También funciona como una de las fuentes de Power BI. 
•	EDA.py: Realiza un análisis EDA del dataset original; analizando cada variable a nivel descriptivo.
•	Avances.xlsx: Muestra una comparación de algunos modelos empleando el dataset original, aunque estos no muestras resultados favorables en esta herramienta.
•	raw_sales.csv: Archivo fuente para los análisis y modelos desarrollados en python. Este archivo solo tiene las columnas de datesold y Price.
•	regresionesmensuales.py: Archivo de python preliminar en que se agrupan las ventas a nivel mensual para realizar los análisis de cada modelo
•	regresiones_trimestrales_final.py: Archivo de Python que agrupa y transforma los datos de raw_sales.csv, realiza los modelos predictivos, realiza gráficas de cada modelo, calcula las métricas de evaluación de ajuste (RSME, MAD, R2) y exporta todo en dos archivos distintos csv.
•	predicciones.csv: Archivo que se origina del script de Python. Contiene todas las predicciones para cada modelo a nivel trimestral.
•	Métricas_de_errores.csv: Contiene los valores RSME, MAD, R2 para cada modelo.
•	Modelos Predictivos.pbix: Un archivo de Power BI que consolida tanto los resultados de las predicciones, como las métricas de errores, para poder compara de forma interactiva los resultados de cada modelo y determinar cual es el mejor modelo para el dataset. 
