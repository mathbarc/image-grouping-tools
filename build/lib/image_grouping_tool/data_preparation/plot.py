from typing import List, Optional
import numpy
import matplotlib.pyplot as plt
import os


def scatterplot_samples(
    feature_vector: numpy.ndarray,
    model_id,
    kept_var: float,
    paths: List[str],
    graph_path: str,
    cluster_ids: Optional[List[int]] = None,
):
    if cluster_ids is not None:
        color_map = plt.cm.get_cmap("hsv")
        cluster_ids = (cluster_ids - min(cluster_ids)) / (
            max(cluster_ids) - min(cluster_ids)
        )

        print(cluster_ids)
    plt.figure(figsize=(8, 6))
    for i, data in enumerate(feature_vector):
        if cluster_ids is None:
            plt.scatter(data[0], data[1], c="green")
        else:
            plt.scatter(data[0], data[1], c=color_map(cluster_ids[i]))

    if feature_vector.shape[0] == len(paths) and len(paths) < 30:
        for i, path in enumerate(paths):
            name = os.path.basename(path)
            name = os.path.splitext(name)[0]
            plt.annotate(name, (feature_vector[i, 0], feature_vector[i, 1]))

    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title(
        f"PCA of {model_id} Embeddings ( kept variance: {round(kept_var * 100)}% )"
    )
    plt.savefig(graph_path + ".png")
    plt.show()
