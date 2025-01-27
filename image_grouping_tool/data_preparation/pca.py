import numpy

from sklearn.decomposition import PCA


def calculate_var(data: numpy.ndarray):
    mean_val = numpy.mean(data, axis=0)
    return numpy.mean(numpy.sqrt(numpy.pow(data - mean_val, 2).sum()))


def pca(feature_vector: numpy.ndarray, n_components: int) -> (numpy.ndarray, float):
    orig_var = calculate_var(feature_vector)
    pca = PCA(n_components=n_components)
    final_features = pca.fit_transform(feature_vector)
    final_var = calculate_var(final_features)
    return final_features, final_var / orig_var
