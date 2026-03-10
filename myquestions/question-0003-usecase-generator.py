import random
import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV

def generar_caso_de_uso_optimizar_modelo_bioinformatica():
    # Parámetros aleatorios para el dataset
    n_samples = random.randint(30, 80)         # pocas muestras (problema bioinformático)
    n_features = random.randint(100, 300)      # muchas variables (genes)
    n_informative = random.randint(10, 30)     # genes realmente relevantes
    noise = round(random.uniform(0.5, 5.0), 2)
    random_state = random.randint(0, 9999)

    # Generar datos sintéticos
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        noise=noise,
        random_state=random_state
    )

    # Calcular la salida esperada (ejecutar la función real)
    model = ElasticNet()
    param_grid = {
        'alpha': [0.1, 1.0, 10.0],
        'l1_ratio': [0.2, 0.5, 0.8]
    }
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='r2')
    grid_search.fit(X, y)

    mejor_r2 = round(grid_search.best_score_, 6)
    mejores_params = grid_search.best_params_

    # Argumentos del caso de uso
    argumentos = {
        'X': X,
        'y': y
    }

    # Salida esperada
    salida_esperada = (mejor_r2, mejores_params)

    return (argumentos, salida_esperada)

