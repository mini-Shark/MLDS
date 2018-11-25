import collections

import pandas as pd
import torch.utils.data as torch_data
from torchvision import transforms
from PIL import Image


class CustomCifar10(torch_data.Dataset):

  def __init__(self, csv_path):
    '''
    csv_path: file of store path-label 
    '''
    csv_data = pd.read_csv(csv_path)
    data=collections.namedtuple('data',['path','label'])
    self.data_pair = [data(path=pair[1]['path'], label=pair[1]['label']) for pair in csv_data.iterrows()] 
    self.to_tensor = transforms.ToTensor() 
    # Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    # [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    self.data_len = len(self.data_pair)

  def __getitem__(self, index):
    img_path, label = self.data_pair[index].path, self.data_pair[index].label
    img_raw = Image.open(img_path)
    img_tensor = self.to_tensor(img_raw)

    return (img_tensor, label)
    
  def __len__(self):
    return self.data_len


