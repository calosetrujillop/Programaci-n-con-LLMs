import pandas as pd
import numpy as np
import random
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA


def generar_caso_de_uso_analizar_eficiencia_energetica():
    """
    Genera un caso de uso aleatorio para la función analizar_eficiencia_energetica(df, target_col).

    Retorna:
        tuple: (
            dict con los argumentos de la función ({"df": DataFrame, "target_col": str}),
            tuple con el resultado esperado (matriz_transformada, pca_entrenado)
        )
    """

    # --- 1. Generación aleatoria del DataFrame de entrada ---
    
    n_edificios = random.randint(50, 150)
    
    # Generar datos base con distribución normal
    np.random.seed(random.randint(1, 10000))
    
    datos = {
        'edificio_id': [f'ED_{str(i+1).zfill(3)}' for i in range(n_edificios)],
        'consumo_kwh': np.random.normal(1200, 300, n_edificios),
        'temperatura_exterior': np.random.normal(22, 8, n_edificios),
        'humedad_relativa': np.random.normal(65, 15, n_edificios),
        'ocupacion_personas': np.random.normal(50, 20, n_edificios),
        'area_m2': np.random.normal(2000, 500, n_edificios),
        'antiguedad_anos': np.random.uniform(1, 30, n_edificios),
        'eficiencia_hvac': np.random.normal(0.75, 0.15, n_edificios),
        'iluminacion_led_pct': np.random.uniform(0, 100, n_edificios),
        'aislamiento_score': np.random.normal(7, 2, n_edificios),
        'ventanas_doble_cristal': np.random.choice([0, 1], n_edificios, p=[0.3, 0.7])
    }
    
    # Añadir algunos outliers para probar robustez
    n_outliers = random.randint(3, 8)
    outlier_indices = random.sample(range(n_edificios), n_outliers)
    
    for idx in outlier_indices:
        datos['consumo_kwh'][idx] *= random.uniform(3, 6)  # Picos de consumo
        if random.random() < 0.5:
            datos['temperatura_exterior'][idx] *= random.uniform(2, 3)
    
    # Introducir valores faltantes aleatoriamente
    columnas_numericas = ['consumo_kwh', 'temperatura_exterior', 'humedad_relativa', 
                         'ocupacion_personas', 'area_m2', 'antiguedad_anos', 
                         'eficiencia_hvac', 'iluminacion_led_pct', 'aislamiento_score']
    
    for col in columnas_numericas:
        if random.random() < 0.7:  # 70% chance de tener algunos NaN
            n_nan = random.randint(1, max(1, n_edificios // 20))
            nan_indices = random.sample(range(n_edificios), n_nan)
            for idx in nan_indices:
                datos[col][idx] = np.nan
    
    df_entrada = pd.DataFrame(datos)
    target_col = 'consumo_kwh'
    
    # --- 2. Cálculo del resultado esperado (réplica la lógica de analizar_eficiencia_energetica) ---
    
    df_proc = df_entrada.copy()
    
    # Paso 1: Identificar columnas numéricas (excluyendo target y columnas no numéricas)
    columnas_numericas = df_proc.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in columnas_numericas:
        columnas_numericas.remove(target_col)
    
    # Extraer características numéricas
    X = df_proc[columnas_numericas]
    
    # Paso 2: SimpleImputer con mediana
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    
    # Paso 3: RobustScaler
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    
    # Paso 4: PCA con 3 componentes
    pca = PCA(n_components=3)
    X_transformed = pca.fit_transform(X_scaled)
    
    # --- 3. Construcción de la tupla de salida ---
    argumentos = {
        "df": df_entrada,
        "target_col": target_col
    }
    
    resultado_esperado = (X_transformed, pca)
    
    return (argumentos, resultado_esperado)


def mostrar_informacion_caso(args, resultado_esperado):
    """
    Función auxiliar para mostrar información del caso de uso generado.
    """
    df = args["df"]
    target_col = args["target_col"]
    matriz_transformada, pca_obj = resultado_esperado
    
    print(f"=== INFORMACIÓN DEL CASO DE USO ===")
    print(f"Forma del DataFrame: {df.shape}")
    print(f"Columna target: {target_col}")
    print(f"Columnas numéricas (sin target): {df.select_dtypes(include=[np.number]).columns.drop(target_col).tolist()}")
    print(f"\n=== VALORES FALTANTES POR COLUMNA ===")
    missing_info = df.isnull().sum()
    missing_info = missing_info[missing_info > 0]
    for col, count in missing_info.items():
        print(f"{col}: {count} valores faltantes")
    
    print(f"\n=== RESULTADO ESPERADO ===")
    print(f"Forma de la matriz transformada: {matriz_transformada.shape}")
    print(f"Varianza explicada por componente: {pca_obj.explained_variance_ratio_.round(3)}")
    print(f"Varianza total explicada: {pca_obj.explained_variance_ratio_.sum():.3f}")
    
    print(f"\n=== PRIMERAS 5 FILAS DE DATOS ORIGINALES ===")
    print(df.head().to_string())
    
    print(f"\n=== PRIMERAS 5 FILAS TRANSFORMADAS ===")
    print(pd.DataFrame(matriz_transformada[:5], 
                      columns=[f'PC{i+1}' for i in range(3)]).round(3).to_string())
