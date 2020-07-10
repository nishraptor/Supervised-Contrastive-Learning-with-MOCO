import torch
import torchvision.datasets as datasets
data = datasets.VOCDetection('datasets/',year='2007',image_set='val', download=True)
