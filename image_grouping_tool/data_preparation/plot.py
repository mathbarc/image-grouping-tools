from typing import List, Optional
import numpy
import os
import plotly.graph_objects
import plotly.offline
import plotly.colors


def scatterplot_samples(
    feature_vector: numpy.ndarray,
    model_id,
    kept_var: float,
    paths: List[str],
    graph_path: str,
    cluster_ids: Optional[List[int]] = None,
):
    cluster_ids_value = None
    if cluster_ids is not None:
        min_cid = min(cluster_ids)
        max_cid = max(cluster_ids)

        if max_cid - min_cid < 1:
            cluster_ids_value = None
        else:
            cluster_ids_value = (cluster_ids - min_cid) / (max_cid - min_cid)

    fig = plotly.graph_objects.Figure()
    fig.update_layout(
        title=plotly.graph_objects.layout.Title(
            text=f"PCA of {model_id} Embeddings ( kept variance: {round(kept_var * 100)}% )"
        ),
        xaxis=plotly.graph_objects.layout.XAxis(
            title=plotly.graph_objects.layout.xaxis.Title(text="Principal Component 1")
        ),
        yaxis=plotly.graph_objects.layout.YAxis(
            title=plotly.graph_objects.layout.yaxis.Title(text="Principal Component 2")
        ),
    )
    if cluster_ids_value is None:
        fig.add_scatter(
            x=feature_vector[:, 0], y=feature_vector[:, 1], mode="markers", text=paths
        )
    else:
        texts = [
            f"{path} ({cluster_id})" for path, cluster_id in zip(paths, cluster_ids)
        ]

        fig.add_scatter(
            x=feature_vector[:, 0],
            y=feature_vector[:, 1],
            mode="markers",
            marker_color=cluster_ids_value,
            marker_colorscale=plotly.colors.sequential.Rainbow,
            text=texts,
        )

    plotly.offline.plot(fig, filename=graph_path + ".html")


def plot_diverse_images(pca_features, data, save_path: str):
    centers = pca_features[data["diverse_images"]]
    fig = plotly.graph_objects.Figure()
    fig.add_scatter(x=pca_features[:,0], y=pca_features[:,1], mode="markers", name="Images")
    fig.add_scatter(x=centers[:,0], y=centers[:,1], mode="markers", name="More Diverse Images")
    plotly.offline.plot(fig, save_path)
