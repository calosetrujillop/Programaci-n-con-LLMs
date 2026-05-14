import numpy as np
from sklearn.decomposition import PCA

def optimizar_pca_señales(X, varianza_objetivo):
    pca = PCA().fit(X)
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    n_comp = np.argmax(cum_var >= varianza_objetivo) + 1
    return (int(n_comp), cum_var)
