import re
import random
import numpy as np
from paddle.fluid.dataloader import dataset
from paddle.io import Dataset
from pymongo import MongoClient
from datetime import timedelta


def charCount(string,char):
  count = 0
  for c in string:
      if c == char:
          count+=1
  return count

class TmallDataset(Dataset):
  def __init__(self, len_seq, unknown_seq, host="localhost", port=27017):
    self.len_seq = len_seq # 指定数据集序列的长度
    self.unknown_seq = unknown_seq # 分割预测的序列下标
    self.client = MongoClient(host=host,port=port)
    self.data = [item for item in 
                  self.client.Tmall.sku.aggregate([
                    {"$group": {"_id":"$skuId"}},
                    {"$project": {"_id":0,"skuId": "$_id"}}])]
    self.num_sample = len(self.data)

  def __len__(self):
      return self.num_sample

  def __getitem__(self, idx):
    sku = self.data[idx]
    sku = list(self.client.Tmall.sku.find(sku).sort("date",-1))
    if len(sku)>self.len_seq:
      sku = sku[:random.randint(self.len_seq,len(sku))]
    sku_seq= self.parse_sku(sku)
    if len(sku_seq) > self.len_seq:
      sku_seq = sku_seq[-self.len_seq:]
    seq = np.array(sku_seq[:-self.unknown_seq]).astype('float32')
    label = np.array(sku_seq)[:,0].astype('float32')
    return seq, label

  def parse_sku(self, sku):
    start_date = sku[-1]['date']
    end_date = sku[0]['date']
    span_day = (end_date - start_date).days
    
    size_feature = self.parse_size(sku[0]['skuName'])
    amt = self.parse_Amt(sku)
    sku_seq = []
    for row in sku:
      assert end_date >= row['date']
      while end_date != row['date']:
        sku_seq.append([0,0,amt,0,size_feature])
        end_date-=timedelta(days=1)
      
      sku_seq.append([row['payItmCnt'], row['payByrCnt'], amt, row['cartCnt'],size_feature])
      end_date-=timedelta(days=1)
    assert len(sku_seq) >= span_day
    while len(sku_seq) < self.len_seq: # 如果现有数据的天数范围不足以充满序列，就再前面填充-1值
      sku_seq.append([-1,-1,amt,-1,size_feature])
    sku_seq.reverse()
    return sku_seq

  def parse_size(self, size):
    size_str = re.findall("尺.:(.+?)($|/| )",size.strip().upper())
    if len(size_str)!=1:
        raise KeyError("未知的尺码",size)
        # print("未知的尺码",size)
        return -1
    size_str = size_str[0][0]
    size = re.findall("(.+?)XL",size_str)
    if len(size)==1:
        if "X" in size[0][0]:
            size = charCount(size_str,"X")+2
        else:
            size = int(size[0])+2
    elif size_str == "XL":
        size = 3
    elif size_str == "L":
        size = 2
    elif size_str == "M":
        size = 1
    elif size_str == "S":
        size = 0
    elif "2" in size_str:
        size = int(size_str[1:])
    elif "均" in size_str:
        size = 4
    else:
        raise KeyError("未知的尺码",size_str)
    return size

  def parse_Amt(self, sku):
    """计算这段时间的Amt成交均价"""
    amt,cnt = 0,0
    for row in sku:
      amt += row['payAmt']
      cnt += row['payItmCnt']
    if cnt>0:
      return amt/cnt
    else:
      return -1

if __name__ == "__main__":
  from paddle.io import DataLoader
  dataset = TmallDataset(10,3)
  # print(dataset[0])
  dataloader = DataLoader(dataset,batch_size=4,shuffle=True,drop_last=True)
  for seq,label in dataloader():
    print(seq.shape, label.shape)