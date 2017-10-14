from torch.utils.data.dataset import Dataset
from torchvision import transforms, datasets
import pandas as pd
import os

data_transforms = transforms.Compose([
    transforms.ToTensor()
])

cyrilic_dataset = datasets.ImageFolder(root='/opt/ProjectsPy/0_DATASETS/Cyrillic-small',
                                       transform=data_transforms)