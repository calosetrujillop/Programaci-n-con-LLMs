import random
import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import cross_val_score


def generar_caso_de_uso_entrenar_regresor_logaritmico():
    # Parámetros aleatorios
    n_samples = random.randint(80, 200)
    n_features = random.randint(5, 20)
    n_informative = random.randint(3, n_features)
    noise = round(random.uniform(0.1, 1.5), 2)
    random_state = random.randint(0, 9999)

    # Generar X e y con valores positivos (costos de infraestructura)
    X, y_raw = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        noise=noise,
        random_state=random_state
    )
    # Transformar y para simular costos con cola larga (siempre positivos)
    y = np.abs(y_raw) + 1.0

    # Calcular salida esperada
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('ridge', Ridge(alpha=1.0))
    ])
    modelo_compuesto = TransformedTargetRegressor(
        regressor=pipeline,
        func=np.log1p,
        inverse_func=np.expm1
    )
    scores = cross_val_score(modelo_compuesto, X, y, cv=5, scoring='r2')
    modelo_compuesto.fit(X, y)

    salida_esperada = {
        'modelo_compuesto': modelo_compuesto,
        'r2_promedio': round(float(np.mean(scores)), 6),
        'n_caracteristicas': X.shape[1]
    }

    argumentos = {'X': X, 'y': y}

    return (argumentos, salida_esperada)

