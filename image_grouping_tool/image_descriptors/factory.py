from . import efficientnet_v2
from . import resnet

from typing import Tuple
import torch
import torchvision


def build_model(
    model_identifier: str,
) -> Tuple[torch.nn.Sequential, int, torchvision.transforms.Compose]:
    if model_identifier == "resnet50":
        descriptor, feature_size = resnet.resnet50_descriptor()
        return descriptor, feature_size, resnet.resnet50_transform()
    elif model_identifier == "efficientnet_v2_m":
        descriptor, feature_size = efficientnet_v2.efficientnet_v2_m_descriptor()
        return descriptor, feature_size, efficientnet_v2.efficientnet_v2_m_transform()
    else:
        raise Exception(f"Invalid model {model_identifier}")
