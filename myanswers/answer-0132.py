import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score

def validar_estratificado(X, y, n_folds, tipo_modelo):
    """
    Realiza validación cruzada estratificada y devuelve métricas de rendimiento.

    Parámetros:
        X          : array numpy de forma (m, n) con las características.
        y          : array numpy de forma (m,) con las etiquetas de clase.
        n_folds    : entero >= 2, número de particiones de la validación cruzada.
        tipo_modelo: string que indica el clasificador a usar.
                     Valores posibles: 'logistic', 'arbol', 'knn'.

    Retorna:
        dict: Diccionario con métricas de rendimiento:
              'metricas_por_fold' : np.ndarray de forma (n_folds, 3)
              'media_accuracy'    : float
              'media_f1'          : float
              'media_precision'   : float
    """
    # 1. Instanciar el modelo
    _modelos = {
        "logistic": LogisticRegression(max_iter=1000),
        "arbol":    DecisionTreeClassifier(random_state=42),
        "knn":      KNeighborsClassifier(n_neighbors=5),
    }
    if tipo_modelo not in _modelos:
        raise ValueError("tipo_modelo no válido. Use 'logistic', 'arbol' o 'knn'.")
    modelo = _modelos[tipo_modelo]

    # 2. Crear un StratifiedKFold
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    # Lista para almacenar las métricas de cada fold
    metricas_por_fold_list = []

    # 3. Implementar manualmente el bucle de validación cruzada
    for train_idx, test_idx in skf.split(X, y):
        # Dividir X e y usando dichos índices
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Entrenar el modelo
        modelo.fit(X_train, y_train)

        # Predecir sobre el subconjunto de prueba
        y_pred = modelo.predict(X_test)

        # Calcular las tres métricas
        accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        precision_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)

        metricas_por_fold_list.append([accuracy, f1_macro, precision_macro])

    # Convertir la lista de métricas a un array numpy
    metricas_por_fold_array = np.array(metricas_por_fold_list)

    # Calcular las medias de las métricas
    media_accuracy = round(float(metricas_por_fold_array[:, 0].mean()), 4)
    media_f1 = round(float(metricas_por_fold_array[:, 1].mean()), 4)
    media_precision = round(float(metricas_por_fold_array[:, 2].mean()), 4)

    # 4. Devolver un diccionario con las cuatro claves
    return {
        'metricas_por_fold': metricas_por_fold_array,
        'media_accuracy': media_accuracy,
        'media_f1': media_f1,
        'media_precision': media_precision
    }
