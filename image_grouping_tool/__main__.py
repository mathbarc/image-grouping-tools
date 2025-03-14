# from image_grouping_tool import __version__
from image_grouping_tool.dataset import ImageFolderDataset
from image_grouping_tool.clustering.similarity import get_distances
from image_grouping_tool.clustering.diversity import get_n_diverse_images
from image_grouping_tool.image_descriptors import build_model, generate_feature_list
from image_grouping_tool.data_preparation.plot import plot_diverse_images
from image_grouping_tool.data_preparation import (
    pca,
    scatterplot_samples,
)

import click
import torch
import os
import shutil


@click.group()
# @click.version_option(__version__, message="%(version)s")
def cli():
    pass


@cli.command(name="compute_features")
@click.argument("image_folder", nargs=1, type=str)
@click.option(
    "--batch_size",
    required=False,
    help="Batch size for feature vector inference",
    type=int,
    default=32 if torch.cuda.is_available() else os.cpu_count(),
)
@click.option(
    "--model_id",
    required=False,
    help="Model for image description ( resnet50 | efficientnet_v2_m )",
    default="resnet50",
    type=str,
)
@click.option(
    "--output", required=False, help="Output file", default="./features.pt", type=str
)
def compute_features(image_folder: str, batch_size: int, model_id: str, output: str):
    model, _, transform = build_model(model_id)
    image_data = ImageFolderDataset(image_folder, transform)
    features = generate_feature_list(image_data, model, batch_size)
    data = {"features": features, "paths": image_data.image_list, "model_id": model_id}
    torch.save(data, output)
    return data


@cli.command(name="apply_pca")
@click.argument("data_file", nargs=1, type=str)
@click.option(
    "--n_components",
    required=False,
    help="Number of desired components on final feature vector",
    type=int,
    default=2,
)
@click.option(
    "--output",
    required=False,
    help="Output file",
    default="./features_pca.pt",
    type=str,
)
@click.option("--not_show", required=False, is_flag=True)
def apply_pca(data_file: str, n_components: int, output: str, not_show: bool):
    data = torch.load(data_file, weights_only=False)

    features, kept_variance = pca(data["features"].numpy(), n_components)
    output_data = {
        "features": features,
        "paths": data["paths"],
        "kept_variance": kept_variance,
        "model_id": data["model_id"],
    }
    torch.save(output_data, output)

    if n_components == 2 and not not_show:
        scatterplot_samples(
            features,
            data["model_id"],
            kept_variance,
            data["paths"],
            os.path.splitext(output)[0],
        )
    else:
        print(f"kept_variance: {kept_variance}")

    return output_data


from sklearn.cluster import DBSCAN, KMeans


@cli.command(name="cluster")
@click.argument("data_file", nargs=1, type=str)
@click.option(
    "--algorithm",
    required=False,
    help="clustering algorithm to be used (dbscan, kmeans)",
    default="dbscan",
    type=str,
)
@click.option(
    "--min_samples",
    required=False,
    help="Minimum number of samples on a neighborhood of a core point, used by dbscan algorithm",
    default=3,
    type=int,
)
@click.option(
    "--eps",
    required=False,
    help="Size of the neighborhood arround each sample, used by dbscan algorithm",
    default=2.0,
    type=float,
)
@click.option(
    "--n_clusters",
    required=False,
    help="number of desired clusters, used by kmeans",
    default=3,
    type=int,
)
@click.option("--not_show", required=False, is_flag=True)
def cluster(
    data_file: str,
    algorithm: str,
    min_samples: int,
    eps: float,
    n_clusters: int,
    not_show: bool,
):
    data = torch.load(data_file, weights_only=False)
    if algorithm == "dbscan":
        cluster_alg = DBSCAN(min_samples=min_samples, eps=eps)
    elif algorithm == "kmeans":
        cluster_alg = KMeans(n_clusters)
    else:
        raise Exception(f"Invalid clustering algorithm: {algorithm}")
    result = cluster_alg.fit_predict(data["features"])
    out_path = os.path.splitext(data_file)[0] + "_cluster"
    data["clusters"] = result
    torch.save(data, out_path + ".pt")
    if data["features"].shape[1] == 2 and not not_show:
        scatterplot_samples(
            data["features"],
            data["model_id"],
            data["kept_variance"],
            data["paths"],
            out_path,
            result,
        )


@cli.command(name="compute_distances")
@click.argument("features_path", nargs=1, type=str)
@click.option("--images", "-i", type=(str), multiple=True, required=True)
@click.option(
    "--dist",
    "-d",
    required=False,
    help="Distance to compute similary: euclidian[default], cosine, minkowski or mahalanobis",
    default="euclidian",
    type=str,
)
@click.option(
    "--output",
    "-o",
    required=False,
    help="Output file",
    default="./features.pt",
    type=str,
)
def compute_distances(features_path: str, images: tuple, dist: str, output: str):
    data = torch.load(features_path, weights_only=False)
    distances, sorted_idx = get_distances(data, images, dist)
    data[f"{dist}-distances"] = distances
    data[f"{dist}-reference_images"] = images
    data[f"{dist}-sorted_idx"] = sorted_idx
    torch.save(data, output)
    return data


@cli.command(name="get_diversity")
@click.argument("features_path", nargs=1, type=str)
@click.option(
    "--number",
    "-n",
    type=(int),
    multiple=False,
    required=True,
    help="Get the n more diverse samples on Dataset.",
)
@click.option(
    "--distance",
    "-d",
    type=(str),
    multiple=False,
    default="euclidian",
    required=False,
    help="Distance to compute diversity: euclidian[default], cosine, minkowski or mahalanobis.",
)
@click.option(
    "--output", required=False, help="Output file", default="./features.pt", type=str
)
@click.option(
    "--dest_folder",
    required=False,
    help="Folder to store most diverse images",
    default=None,
    type=str,
)
def get_diversity(
    features_path: str, number: int, distance: str, output: str, dest_folder: str
):
    data = torch.load(features_path, weights_only=False)
    images_idx = get_n_diverse_images(data["features"], number, distance)
    data[f"diverse_images"] = images_idx
    torch.save(data, output)

    if dest_folder is not None:
        if os.path.exists(dest_folder):
            shutil.rmtree(dest_folder)

        os.makedirs(dest_folder, exist_ok=True)

        for idx in images_idx:
            path: str = data["paths"][idx]
            filename = os.path.basename(path)
            dest_path = os.path.join(dest_folder, filename)
            shutil.copy(path, dest_path)

    return data


@cli.command(name="plot_diversity")
@click.argument("features_path", nargs=1, type=str)
@click.argument("save_path", nargs=1, type=str, default="./diversity_plot.html")
def plot_diversity(features_path: str, save_path: str):
    data = torch.load(features_path, weights_only=False)
    pca_features, kept_variance = pca(data["features"].numpy(), 2)
    plot_diverse_images(pca_features, data, save_path)
