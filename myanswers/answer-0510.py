import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity

def estimar_densidad_recorridos(df: pd.DataFrame, duration_col: str, bandwidth: float, num_points: int) -> pd.DataFrame:
    """
    Modela de forma suave la distribución de duraciones de recorridos
    utilizando estimación de densidad de kernel.

    Parámetros:
        df (pd.DataFrame): DataFrame con los datos de recorridos.
        duration_col (str): Nombre de la columna que contiene la duración del viaje en minutos.
        bandwidth (float): Ancho de banda para el modelo KernelDensity.
        num_points (int): Número de puntos para la malla de evaluación de la densidad.

    Retorna:
        pd.DataFrame: DataFrame con las columnas 'duracion' y 'densidad',
                      ordenado de menor a mayor por 'duracion'.
    """
    # 1. Seleccionar la columna de duración y eliminar nulos
    cleaned_series = df[duration_col].dropna()

    # 2. Convertir a numpy array y construir la malla
    training_array = cleaned_series.to_numpy(dtype=float).reshape(-1, 1)
    grid = np.linspace(cleaned_series.min(), cleaned_series.max(), num_points).reshape(-1, 1)

    # 3. Ajustar el modelo KernelDensity y evaluar las densidades
    kde_model = KernelDensity(kernel="gaussian", bandwidth=bandwidth)
    kde_model.fit(training_array)
    log_density = kde_model.score_samples(grid)
    density = np.exp(log_density)

    # 4. Devolver un pd.DataFrame ordenado
    result_df = pd.DataFrame(
        {
            "duracion": grid[:, 0],
            "densidad": density,
        }
    ).sort_values(by="duracion", ascending=True, kind="mergesort").reset_index(drop=True)

    return result_df
