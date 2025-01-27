import torch
import torchvision


def efficientnet_v2_m_descriptor():
    effnet = torchvision.models.efficientnet_v2_m(
        weights=torchvision.models.EfficientNet_V2_M_Weights.IMAGENET1K_V1
    )
    layers = list(effnet.children())
    model = torch.nn.Sequential(*layers[:-1])
    model.add_module("squeeze", torch.nn.Flatten())
    return model.eval(), layers[2][-1].in_features


def efficientnet_v2_m_transform() -> torchvision.transforms.Compose:
    return torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((480, 480)),  # (227,227) for vgg
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )


if __name__ == "__main__":
    model, n_features = efficientnet_v2_m_descriptor()
    print(model)
    print(n_features)
