from numpy import require
from torchvision import models
from image_grouping_tool import __version__
from image_grouping_tool.dataset import ImageFolderDataset
from image_grouping_tool.image_descriptors import build_model, generate_feature_list
from image_grouping_tool.data_preparation import (
    pca,
    scatterplot_samples,
)

import click
import torch
import os


@click.group()
@click.version_option(__version__, message="%(version)s")
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
def apply_pca(data_file: str, n_components: int, output: str):
    data = torch.load(data_file)
    features, kept_variance = pca(data["features"].numpy(), n_components)
    output_data = {
        "features": features,
        "paths": data["paths"],
        "kept_variance": kept_variance,
        "model_id": data["model_id"],
    }

    scatterplot_samples(
        features,
        data["model_id"],
        kept_variance,
        data["paths"],
        os.path.splitext(output)[0],
    )

    torch.save(output_data, output)
    return output_data
