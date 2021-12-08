import random
import numpy as np
from paddle.io import DataLoader, Dataset

class SaleDataset(Dataset):
  def __init__(self, num_sample, normalized=1):
    self.num_sample = num_sample
    self.normalized = normalized
  
  def __getitem__(self, idx):
    i = np.random.randint(0,100)
    if random.random()>0.5:
      num_seq = np.linspace(i,i+14,15,dtype="float32")/self.normalized # 除以均值进行标准化
    else:
      num_seq = np.linspace(i,i-14,15,dtype="float32")/self.normalized # 除以均值进行标准化
    seq = np.expand_dims(num_seq[:-1], axis=0)
    # label = np.expand_dims(num_seq, axis=0)
    label = num_seq
    return seq, label

  def __len__(self):
      return self.num_sample

if __name__ == "__main__":
  dataloader = DataLoader(SaleDataset(10),batch_size=4,shuffle=True,drop_last=True)
  for seq,label in dataloader():
  # for images, labels in SaleDataset(100):
    print(seq.shape, label.shape)