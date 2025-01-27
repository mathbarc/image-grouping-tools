from image_grouping_tool.dataset import ImageFolderDataset
from image_grouping_tool.image_descriptors import build_model, generate_feature_list
from image_grouping_tool.data_preparation import (
    pca,
    scatterplot_samples,
)

if __name__ == "__main__":
    # folder_dir = "/data/ssd1/Datasets/Faces/training/"
    # folder_dir = "/data/ssd1/Datasets/Coco/test2014/"
    # folder_dir = "/data/ssd1/Datasets/Coco/train2014/"
    folder_dir = "/data/ssd1/Datasets/Plantas/"

    # model_id = "resnet50"
    model_id = "efficientnet_v2_m"

    model, n_features, transform = build_model(model_id)
    dataset = ImageFolderDataset(folder_dir, transform)

    features = generate_feature_list(dataset, model)
    final_features, kept_var = pca(features.numpy(), 2)
    print(final_features.shape)
    print(kept_var)

    scatterplot_samples(final_features, model_id, dataset.image_list)
