from typing import Optional
import torch
from torch.utils.data import Dataset

from glob import glob
import torchvision
from torchvision.io import read_image
import os


class ImageFolderDataset(Dataset):
    def __init__(
        self, path: str, transform: Optional[torchvision.transforms.Compose] = None
    ) -> None:
        super().__init__()
        self.image_list = []
        self.image_list.extend(glob("*.png", root_dir=path))
        self.image_list.extend(glob("*.jpeg", root_dir=path))
        self.image_list.extend(glob("*.jpg", root_dir=path))

        self.image_list = [os.path.join(path, img_path) for img_path in self.image_list]
        self.transform = transform

    def __getitem__(self, idx) -> torch.Tensor:
        img = read_image(
            self.image_list[idx], torchvision.io.ImageReadMode.RGB
        ).float() * (1.0 / 255.0)

        if self.transform is not None:
            return self.transform(img)
        else:
            return img

    def __len__(self) -> int:
        return len(self.image_list)


if __name__ == "__main__":
    dataset = ImageFolderDataset("/data/ssd1/Datasets/Plantas/")
    print(dataset.image_list, len(dataset))
    image = dataset[0]
    print(image.size())
