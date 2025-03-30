from torch.cuda import is_available
from . import efficientnet_v2
from . import resnet

from typing import Tuple
import torch
import torchvision

import os
import tqdm
import cpuinfo


def build_model(
    model_identifier: str,
) -> Tuple[torch.nn.Sequential, int, torchvision.transforms.Compose]:
    if model_identifier == "resnet50":
        descriptor, feature_size = resnet.resnet50_descriptor()
        return optimize_model(descriptor), feature_size, resnet.resnet50_transform()
    elif model_identifier == "efficientnet_v2_m":
        descriptor, feature_size = efficientnet_v2.efficientnet_v2_m_descriptor()
        return (
            optimize_model(descriptor),
            feature_size,
            efficientnet_v2.efficientnet_v2_m_transform(),
        )
    else:
        raise Exception(f"Invalid model {model_identifier}")


def optimize_model(model: torch.nn.Sequential):
    if torch.cuda.is_available():
        model = model.to(torch.device("cuda"))
    else:
        model = model.to("cpu")
        vendor = cpuinfo.get_cpu_info()["vendor_id_raw"]
        if vendor == "GenuineIntel":
            import intel_extension_for_pytorch as ipex

            model = ipex.optimize(model)

    return model


def generate_feature_list(
    dataset: torch.utils.data.Dataset,
    descriptor: torch.nn.Module,
    batch_size: int = 32 if torch.cuda.is_available() else os.cpu_count(),
    device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
):
    with torch.no_grad():
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

        feature_list = torch.Tensor()

        for images in tqdm.tqdm(dataloader):
            features = descriptor(images.to(device))
            feature_list = torch.concat([feature_list, features.cpu()])

        return feature_list.cpu()
