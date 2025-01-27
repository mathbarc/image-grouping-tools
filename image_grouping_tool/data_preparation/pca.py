import numpy

from sklearn.decomposition import PCA


def calculate_var(data: numpy.ndarray):
    mean_val = numpy.mean(data, axis=0)
    diff = data - mean_val
    return numpy.mean(numpy.sqrt(numpy.pow(diff, 2).sum()))


def pca(feature_vector: numpy.ndarray, n_components: int) -> (numpy.ndarray, float):
    pca = PCA(n_components=n_components)
    final_features = pca.fit_transform(feature_vector)
    final_var = calculate_var(final_features)
    orig_var = calculate_var(feature_vector)
    return final_features, final_var / orig_var
