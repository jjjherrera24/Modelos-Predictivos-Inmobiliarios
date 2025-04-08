import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, r2_score

def calcular_errores(y_true, y_pred):

    r2 = round(r2_score(y_true, y_pred),4)
    rmse = int(np.sqrt(mean_squared_error(y_true, y_pred)))
    mad = int(np.mean(np.abs(y_true - y_pred)))

    return r2, rmse, mad

def agrupar_por_trimestre(dataset, target_column='price'):
    # Cargar el dataset
    data = pd.read_csv(dataset)
    
    # Limpiar valores nulos en las columnas 'price' y 'datesold'
    data = data.dropna(subset=['price', 'datesold'])
    
    # Convertir la columna 'datesold' a tipo datetime
    data['datesold'] = pd.to_datetime(data['datesold'])
    
    # Agrupar por trimestre
    data['quarter'] = data['datesold'].dt.to_period('Q')
    grouped = data.groupby('quarter').agg({target_column: 'sum'})
    
    print("\nDatos agrupados por trimestre (suma de precios):\n", grouped)
    
    return grouped

def realizar_predicciones_futuras(dataset, target_column='price'):

    # Agrupar los datos por trimestre antes de aplicar los modelos
    data_agrupada = agrupar_por_trimestre(dataset, target_column)
    
    # Preparar los datos para el modelo
    X = np.array(range(len(data_agrupada)))  
    X = X.reshape(-1, 1)  

    y = data_agrupada[target_column].values 

    quarters = data_agrupada.index.astype(str).to_list() 
    
    # Obtener el último trimestre
    last_quarter = data_agrupada.index[-1]
    last_quarter = last_quarter.asfreq('Q-MAR')

    # Generar trimestres futuros
    future_quarters = [last_quarter + pd.offsets.QuarterEnd(i) for i in range(1, 5)]
    future_quarters = [f"{q.year}Q{q.quarter}" for q in future_quarters]
    future_quarters = sorted(future_quarters, key=lambda x: pd.to_datetime(x.split('Q')[0] + '-' + x.split('Q')[1] + '-01'))

    
    # 1. Regresión Lineal
    lin_reg = LinearRegression()
    lin_reg.fit(X, y)
    lin_reg_pred = lin_reg.predict(np.array(range(len(data_agrupada) + 4)).reshape(-1, 1))

    # 2. Regresión Cuadrática
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)
    poly_reg = LinearRegression()
    poly_reg.fit(X_poly, y)
    poly_reg_pred = poly_reg.predict(poly.transform(np.array(range(len(data_agrupada) + 4)).reshape(-1, 1)))

    # 3. Regresión Exponencial
    X_log = np.log(X + 1) 
    lin_reg_exp = LinearRegression()
    lin_reg_exp.fit(X_log, y)
    lin_reg_exp_pred = lin_reg_exp.predict(np.log(np.array(range(len(data_agrupada) + 4)) + 1).reshape(-1, 1))

    '''
    # 4. Promedio Móvil
    window = 4
    predicciones = 4
    promedio_movil = [math.nan] * 4  
    for i in range(len(y) - window):
        promedio = np.mean(y[i:i+window])
        promedio_movil.append(promedio)
    if len(promedio_movil) > window:
        last_window_average = np.mean(promedio_movil[-window:])  
    for _ in range(predicciones):
        promedio_movil.append(last_window_average)
    '''

    # 5. Holt-Winters
    hw_model = ExponentialSmoothing(y, trend='add', seasonal='add', seasonal_periods=4, initialization_method="estimated")
    hw_fit = hw_model.fit()
    hw_fitted_values = hw_fit.fittedvalues
    hw_forecast = hw_fit.forecast(steps = 4)
    hw_result = np.append(hw_fitted_values, hw_forecast)

    # 6. ARIMA
    arima_model = ARIMA(y, order=(1, 1, 1))
    arima_fit = arima_model.fit()
    arima_fitted_values = arima_fit.fittedvalues
    arima_forecast = arima_fit.forecast(steps=4)
    arima_result = np.append(arima_fitted_values, arima_forecast)

    r2_lin, rmse_lin, mad_lin = calcular_errores(y, lin_reg.predict(X))
    r2_poly, rmse_poly, mad_poly = calcular_errores(y, poly_reg.predict(X_poly))
    r2_exp, rmse_exp, mad_exp = calcular_errores(y, lin_reg_exp.predict(np.log(X + 1).reshape(-1, 1)))
    r2_hw, rmse_hw, mad_hw = calcular_errores(y, hw_fitted_values)
    r2_arima, rmse_arima, mad_arima = calcular_errores(y, arima_fitted_values)


    resultados_errores = pd.DataFrame({
        'Modelo': ['Regresion Lineal', 'Regresion Cuadratica', 'Regresion Exponencial', 'Holt-Winters', 'ARIMA'],
        'R^2': [r2_lin, r2_poly, r2_exp, r2_hw, r2_arima],
        'RMSE': [rmse_lin, rmse_poly, rmse_exp, rmse_hw, rmse_arima],
        'MAD': [mad_lin, mad_poly, mad_exp, mad_hw, mad_arima]
    })


    # 1. Regresión Lineal
    plt.figure(figsize=(12, 8))
    plt.plot(quarters, y, label="Valores Reales", color='blue')
    plt.plot(quarters + future_quarters, lin_reg_pred, label="Línea de Tendencia Lineal", color='purple', linestyle='-.')
    plt.title("Regresión Lineal")
    plt.xticks(rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 2. Regresión Cuadrática
    plt.figure(figsize=(12, 8))
    plt.plot(quarters, y, label="Valores Reales", color='blue')
    plt.plot(quarters + future_quarters, poly_reg_pred, label="Línea de Tendencia Cuadrática", color='purple', linestyle='-.')
    plt.title("Regresión Cuadrática")
    plt.xticks(rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 3. Regresión Exponencial
    plt.figure(figsize=(12, 8))
    plt.plot(quarters, y, label="Valores Reales", color='blue')
    plt.plot(quarters + future_quarters, lin_reg_exp_pred, label="Línea de Tendencia Exponencial", color='purple', linestyle='-.')
    plt.title("Regresión Exponencial")
    plt.xticks(rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.show()

    '''
    # 4. Promedio Móvil
    plt.figure(figsize=(12, 8))
    plt.plot(quarters, y, label="Valores Reales", color='blue')
    plt.plot(quarters + future_quarters, promedio_movil, label="Línea de Tendencia Promedio Móvil", color='purple', linestyle='-.')
    plt.title("Promedio Móvil")
    plt.xticks(rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.show()
    '''
    # 5. Holt-Winters
    plt.figure(figsize=(12, 8))
    plt.plot(quarters, y, label="Valores Reales", color='blue')
    plt.plot(quarters + future_quarters, hw_result, label="Línea de Tendencia Holt-Winters", color='purple', linestyle='-.')
    plt.title("Holt-Winters")
    plt.xticks(rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 6. ARIMA
    plt.figure(figsize=(12, 8))
    plt.plot(quarters, y, label="Valores Reales", color='blue')
    plt.plot(quarters + future_quarters, arima_result, label="Línea de Tendencia ARIMA", color='orange', linestyle='--')
    plt.title("Modelo ARIMA")
    plt.xticks(rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.show()




    # Graficar todos los modelos en un gráfico
    plt.figure(figsize=(18, 10))

    # 1. Regresión Lineal
    plt.subplot(2, 3, 1)
    plt.plot(quarters, y, label="Valores Reales", color='blue')
    plt.plot(quarters + future_quarters, lin_reg_pred, label="Regresión Lineal", color='purple', linestyle='-.')
    plt.title("Regresión Lineal")
    plt.xticks([])
    plt.legend()

    # 2. Regresión Cuadrática
    plt.subplot(2, 3, 2)
    plt.plot(quarters, y, label="Valores Reales", color='blue')
    plt.plot(quarters + future_quarters, poly_reg_pred, label="Regresión Cuadrática", color='purple', linestyle='-.')
    plt.title("Regresión Cuadrática")
    plt.xticks([])
    plt.legend()

    # 3. Regresión Exponencial
    plt.subplot(2, 3, 3)
    plt.plot(quarters, y, label="Valores Reales", color='blue')
    plt.plot(quarters + future_quarters, lin_reg_exp_pred, label="Regresión Exponencial", color='purple', linestyle='-.')
    plt.title("Regresión Exponencial")
    plt.xticks([])
    plt.legend()

    '''
    # 4. Promedio Móvil
    plt.subplot(2, 3, 4)
    plt.plot(quarters, y, label="Valores Reales", color='blue')
    plt.plot(quarters + future_quarters, promedio_movil, label="Línea de Tendencia Promedio Móvil", color='purple', linestyle='-.')
    plt.title("Promedio Móvil")
    plt.xticks([])
    plt.legend()
    '''

    # 5. Holt-Winters
    plt.subplot(2, 3, 4)
    plt.plot(quarters, y, label="Valores Reales", color='blue')
    plt.plot(quarters + future_quarters, hw_result, label="Holt-Winters", color='purple', linestyle='-.')
    plt.title("Holt-Winters")
    plt.xticks([])
    plt.legend()

    # 6. ARIMA
    plt.subplot(2, 3, 5)
    plt.plot(quarters, y, label="Valores Reales", color='blue')
    plt.plot(quarters + future_quarters, arima_result, label="ARIMA", color='orange', linestyle='--')
    #plt.plot(future_quarters, forecast_arima, label="Predicciones Futuras ARIMA", color='green', linestyle='-.')
    plt.title("Modelo ARIMA")
    plt.xticks([])
    plt.legend()

    # 6. Tabla
    plt.subplot(2, 3, 6)
    table = plt.table(cellText=resultados_errores.values, colLabels=["Modelo", "R²", "RMSE", "MAD"], loc="center", cellLoc="center", colColours=["#f5f5f5"]*4)
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    # Crear un DataFrame con las predicciones para exportar
    predicciones_df = pd.DataFrame({
        'Trimestre': quarters + future_quarters,
        'Valores Reales': np.append(y, [np.nan] * 4),
        'Regresion Lineal': lin_reg.predict(np.array(range(len(data_agrupada) + 4)).reshape(-1, 1)),
        'Regresion Cuadratica': poly_reg.predict(poly.transform(np.array(range(len(data_agrupada) + 4)).reshape(-1, 1))),
        'Regresion Exponencial': lin_reg_exp.predict(np.log(np.array(range(len(data_agrupada) + 4)) + 1).reshape(-1, 1)),
        'Holt-Winters': hw_result,
        'ARIMA': arima_result
    })

    # Exportar las predicciones a un archivo CSV
    predicciones_df.to_csv('lineas_tendencia_completas.csv', index=False)
    print("\nLas líneas de tendencia se han exportado a 'lineas_tendencia_completas.csv'.")
    resultados_errores.to_csv('metricas_de_errores.csv', index=False)
    print("\nLas metricas de errores se han exportado a 'metricas_de_errores.csv'.")
    return lin_reg_pred, poly_reg_pred, lin_reg_exp_pred, hw_result, arima_result

# Uso del algoritmo
dataset = 'C:\\Users\\jjjhe\\OneDrive\\Desktop\\Maestría Analítica de Datos\\Modelos Predictivos\\recursos\\raw_sales.csv'
resultados = realizar_predicciones_futuras(dataset, target_column='price')
