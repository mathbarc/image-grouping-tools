from typing import List
import numpy
import matplotlib.pyplot as plt
import os


def scatterplot_samples(feature_vector: numpy.ndarray, model_id, paths: List[str]):
    plt.figure(figsize=(8, 6))
    plt.scatter(feature_vector[:, 0], feature_vector[:, 1])

    if feature_vector.shape[0] == len(paths) and len(paths) < 30:
        for i, path in enumerate(paths):
            name = os.path.basename(path)
            name = os.path.splitext(name)[0]
            plt.annotate(name, (feature_vector[i, 0], feature_vector[i, 1]))

    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title(f"PCA of {model_id} Embeddings")
    plt.show()
