from typing import List
import numpy
import torch
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from image_grouping_tool.dataset import ImageFolderDataset
from image_grouping_tool.image_descriptors import build_model

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

import os


def generate_feature_list(
    dataset: torch.utils.data.Dataset,
    descriptor: torch.nn.Module,
    batch_size: int = 32,
    device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
):
    with torch.no_grad():
        descriptor = descriptor.to(device)
        dataloader = DataLoader(dataset, batch_size=batch_size)

        feature_list = torch.Tensor()

        for images in dataloader:
            features = descriptor(images.to(device))
            feature_list = torch.concat([feature_list, features.cpu()])

        return feature_list.cpu()


def pca(feature_vector: numpy.ndarray, n_components: int) -> numpy.ndarray:
    pca = PCA(n_components=n_components)
    final_features = pca.fit_transform(feature_vector)
    return final_features


def scatterplot_samples(feature_vector: numpy.ndarray, model_id, paths: List[str]):
    plt.figure(figsize=(8, 6))
    plt.scatter(feature_vector[:, 0], feature_vector[:, 1])

    if feature_vector.shape[0] == len(paths):
        for i, path in enumerate(paths):
            plt.annotate(
                os.path.basename(path), (feature_vector[i, 0], feature_vector[i, 1])
            )

    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title(f"PCA of {model_id} Embeddings")
    plt.show()


if __name__ == "__main__":
    folder_dir = "/data/ssd1/Datasets/Plantas/"
    # folder_dir = "/data/ssd1/Datasets/Plantas/"

    # model_id = "resnet50"
    model_id = "efficientnet_v2_m"

    model, n_features, transform = build_model(model_id)
    dataset = ImageFolderDataset(folder_dir, transform)

    features = generate_feature_list(dataset, model)
    final_features = pca(features.numpy(), 2)
    print(final_features.shape)

    scatterplot_samples(final_features, model_id, dataset.image_list)
