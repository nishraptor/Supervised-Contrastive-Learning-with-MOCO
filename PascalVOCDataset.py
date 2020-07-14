import torch
import numpy as np
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset


class PascalVOCDataset(Dataset):

    def __init__(self, csv_file, transform=None):
        self.csv_file = csv_file
        self.transform = transform

        self.image_list = pd.read_csv(csv_file)

    def __getitem__(self, i):
        image = Image.open(self.image_list['image'][i])
        label = self.image_list['label'][i]

        if self.transform:
            image = self.transform(image)

        label = torch.from_numpy(np.asarray([label]))
        return image, label

    def __len__(self):
        return len(self.image_list)