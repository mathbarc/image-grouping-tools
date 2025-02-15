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
    if cluster_ids is not None:
        min_cid = min(cluster_ids)
        max_cid = max(cluster_ids)

        if max_cid - min_cid < 1:
            cluster_ids = None
        else:
            cluster_ids = (cluster_ids - min_cid) / (max_cid - min_cid)

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
    if cluster_ids is None:
        fig.add_scatter(
            x=feature_vector[:, 0], y=feature_vector[:, 1], mode="markers", text=paths
        )
    else:
        fig.add_scatter(
            x=feature_vector[:, 0],
            y=feature_vector[:, 1],
            mode="markers",
            marker_color=cluster_ids,
            marker_colorscale=plotly.colors.sequential.Rainbow,
            text=paths,
        )

    plotly.offline.plot(fig, filename=graph_path + ".html")
