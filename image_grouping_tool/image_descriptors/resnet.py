import torch
import torchvision


def resnet50_descriptor(
    weights: torchvision.models.ResNet50_Weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1,
):
    resnet = torchvision.models.resnet50(weights=weights)
    layers = list(resnet.children())
    model = torch.nn.Sequential(*layers[:-1])
    model.add_module("squeeze", torch.nn.Flatten())
    return model.eval(), resnet.fc.in_features


def resnet50_transform() -> torchvision.transforms.Compose:
    return torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((224, 224)),  # (227,227) for vgg
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )


if __name__ == "__main__":
    model, n_features = resnet50_descriptor()
    print(model)
    print(f"output features: {n_features}")
