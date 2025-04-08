import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def agrupar_por_mes(dataset, target_column='price'):
    # Cargar el dataset
    data = pd.read_csv(dataset)
    
    # Convertir la columna 'datesold' a tipo datetime
    if 'datesold' in data.columns:
        data['datesold'] = pd.to_datetime(data['datesold'])
    
    # Agrupar por mes y sumar los valores de 'price'
    data['month'] = data['datesold'].dt.to_period('M')  # Agrupar por mes
    grouped = data.groupby('month').agg({target_column: 'sum'})  # Suma de precios por mes
    
    print("\nDatos agrupados por mes (suma de precios):\n", grouped)
    
    return grouped

def analizar_modelos_regresion(dataset, target_column='price'):
    # Agrupar los datos por mes antes de aplicar los modelos
    data_agrupada = agrupar_por_mes(dataset, target_column)
    
    # Preparar los datos para el modelo
    X = np.array(range(len(data_agrupada)))  # Usamos el índice como variable predictora
    X = X.reshape(-1, 1)  # Convertir a matriz columna para sklearn
    y = data_agrupada[target_column].values  # Los valores de precios

    # 3. Separar el dataset en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Inicializar una lista para almacenar los resultados
    results = []

    # 4. Regresión Lineal
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)
    y_pred_lin = lin_reg.predict(X_test)
    rmse_lin = np.sqrt(mean_squared_error(y_test, y_pred_lin))
    r2_lin = r2_score(y_test, y_pred_lin)
    results.append(["Regresión Lineal", rmse_lin, r2_lin])
    
    # Mostrar gráfico con RMSE y R² en el título
    plt.figure(figsize=(8, 6))
    plt.plot(y_test, label="Valores Reales", color='blue')
    plt.plot(y_pred_lin, label="Predicción Lineal", color='red', linestyle='--')
    plt.title(f"Regresión Lineal - RMSE: {rmse_lin:.4f}, R²: {r2_lin:.4f}")
    plt.legend()
    plt.show()

    # 5. Regresión Cuadrática
    poly = PolynomialFeatures(degree=2)
    X_poly_train = poly.fit_transform(X_train)
    X_poly_test = poly.transform(X_test)
    poly_reg = LinearRegression()
    poly_reg.fit(X_poly_train, y_train)
    y_pred_poly = poly_reg.predict(X_poly_test)
    rmse_poly = np.sqrt(mean_squared_error(y_test, y_pred_poly))
    r2_poly = r2_score(y_test, y_pred_poly)
    results.append(["Regresión Cuadrática", rmse_poly, r2_poly])
    
    # Mostrar gráfico con RMSE y R² en el título
    plt.figure(figsize=(8, 6))
    plt.plot(y_test, label="Valores Reales", color='blue')
    plt.plot(y_pred_poly, label="Predicción Cuadrática", color='green', linestyle='--')
    plt.title(f"Regresión Cuadrática - RMSE: {rmse_poly:.4f}, R²: {r2_poly:.4f}")
    plt.legend()
    plt.show()

    # 6. Regresión Exponencial (modelado logarítmico)
    X_train_log = np.log(X_train + 1)  # Para evitar log(0)
    X_test_log = np.log(X_test + 1)
    exp_reg = LinearRegression()
    exp_reg.fit(X_train_log, y_train)
    y_pred_exp = exp_reg.predict(X_test_log)
    rmse_exp = np.sqrt(mean_squared_error(y_test, y_pred_exp))
    r2_exp = r2_score(y_test, y_pred_exp)
    results.append(["Regresión Exponencial", rmse_exp, r2_exp])
    
    # Mostrar gráfico con RMSE y R² en el título
    plt.figure(figsize=(8, 6))
    plt.plot(y_test, label="Valores Reales", color='blue')
    plt.plot(y_pred_exp, label="Predicción Exponencial", color='orange', linestyle='--')
    plt.title(f"Regresión Exponencial - RMSE: {rmse_exp:.4f}, R²: {r2_exp:.4f}")
    plt.legend()
    plt.show()

    # 7. Promedio Estático (Media)
    media = y_train.mean()
    y_pred_media = np.full_like(y_test, media)
    rmse_media = np.sqrt(mean_squared_error(y_test, y_pred_media))
    r2_media = r2_score(y_test, y_pred_media)
    results.append(["Promedio Estático", rmse_media, r2_media])
    
    # Mostrar gráfico con RMSE y R² en el título
    plt.figure(figsize=(8, 6))
    plt.plot(y_test, label="Valores Reales", color='blue')
    plt.plot(y_pred_media, label="Predicción Promedio Estático", color='purple', linestyle='--')
    plt.title(f"Promedio Estático - RMSE: {rmse_media:.4f}, R²: {r2_media:.4f}")
    plt.legend()
    plt.show()

    # 8. Promedio Móvil
    window = 3
    y_pred_ma = pd.Series(y_test).rolling(window=window).mean().dropna()
    rmse_ma = np.sqrt(mean_squared_error(y_test[window-1:], y_pred_ma))
    r2_ma = r2_score(y_test[window-1:], y_pred_ma)
    results.append(["Promedio Móvil", rmse_ma, r2_ma])
    
    # Mostrar gráfico con RMSE y R² en el título
    plt.figure(figsize=(8, 6))
    plt.plot(y_test, label="Valores Reales", color='blue')
    plt.plot(y_pred_ma, label="Predicción Promedio Móvil", color='brown', linestyle='--')
    plt.title(f"Promedio Móvil - RMSE: {rmse_ma:.4f}, R²: {r2_ma:.4f}")
    plt.legend()
    plt.show()

    # 9. Suavizamiento Exponencial Simple
    ses = ExponentialSmoothing(y_train, trend=None, seasonal=None, initialization_method="estimated")
    ses_model = ses.fit()
    y_pred_ses = ses_model.forecast(len(y_test))
    rmse_ses = np.sqrt(mean_squared_error(y_test, y_pred_ses))
    r2_ses = r2_score(y_test, y_pred_ses)
    results.append(["Suavizamiento Exponencial Simple", rmse_ses, r2_ses])
    
    # Mostrar gráfico con RMSE y R² en el título
    plt.figure(figsize=(8, 6))
    plt.plot(y_test, label="Valores Reales", color='blue')
    plt.plot(y_pred_ses, label="Predicción SES", color='pink', linestyle='--')
    plt.title(f"Suavizamiento Exponencial Simple - RMSE: {rmse_ses:.4f}, R²: {r2_ses:.4f}")
    plt.legend()
    plt.show()

    # 10. Modelo de Holt (Tendencia)
    holt = ExponentialSmoothing(y_train, trend='add', seasonal=None, initialization_method="estimated")
    holt_model = holt.fit()
    y_pred_holt = holt_model.forecast(len(y_test))
    rmse_holt = np.sqrt(mean_squared_error(y_test, y_pred_holt))
    r2_holt = r2_score(y_test, y_pred_holt)
    results.append(["Modelo de Holt", rmse_holt, r2_holt])
    
    # Mostrar gráfico con RMSE y R² en el título
    plt.figure(figsize=(8, 6))
    plt.plot(y_test, label="Valores Reales", color='blue')
    plt.plot(y_pred_holt, label="Predicción Holt", color='yellow', linestyle='--')
    plt.title(f"Modelo de Holt - RMSE: {rmse_holt:.4f}, R²: {r2_holt:.4f}")
    plt.legend()
    plt.show()

    # 11. Modelo de Holt-Winters (Estacionalidad)
    holt_winters = ExponentialSmoothing(y_train, trend='add', seasonal='add', seasonal_periods=12, initialization_method="estimated")
    hw_model = holt_winters.fit()
    y_pred_hw = hw_model.forecast(len(y_test))
    rmse_hw = np.sqrt(mean_squared_error(y_test, y_pred_hw))
    r2_hw = r2_score(y_test, y_pred_hw)
    results.append(["Modelo Holt-Winters", rmse_hw, r2_hw])
    
    # Mostrar gráfico con RMSE y R² en el título
    plt.figure(figsize=(8, 6))
    plt.plot(y_test, label="Valores Reales", color='blue')
    plt.plot(y_pred_hw, label="Predicción Holt-Winters", color='cyan', linestyle='--')
    plt.title(f"Modelo Holt-Winters - RMSE: {rmse_hw:.4f}, R²: {r2_hw:.4f}")
    plt.legend()
    plt.show()

    # Crear el DataFrame con los resultados
    results_df = pd.DataFrame(results, columns=["Modelo", "RMSE", "R²"])
    
    # Imprimir la tabla comparativa
    print("\nTabla Comparativa de RMSE y R²:")
    print(results_df)
    
    return results_df


# Uso del algoritmo
dataset = 'C:\\Users\\jjjhe\\OneDrive\\Desktop\\Maestría Analítica de Datos\\Modelos Predictivos\\recursos\\raw_sales.csv'
analizar_modelos_regresion(dataset, target_column='price')