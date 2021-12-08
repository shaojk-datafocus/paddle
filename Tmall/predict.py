import paddle
from paddle.fluid.layers.nn import pad
from paddle.nn import functional as F
from paddle.io import DataLoader
from easydict import EasyDict
from model import SaleModel
from dataset import SaleDataset
from tqdm import tqdm

def predict(model, dataloader, config, logger=None):
  # 检测是否可以使用GPU，如果可以优先使用GPU
  use_gpu = True if paddle.get_device().startswith("gpu") else False
  if use_gpu:
      paddle.set_device('gpu:0')

  # 指定优化策略，更新模型参数
  optimizer = paddle.optimizer.Adam(learning_rate=config.LEARNING_RATE, beta1=0.9, beta2=0.999, parameters=model.parameters()) 

  # 开启模型预测模式
  model.eval()
  
  i = 0
  for epoch in range(config.EPOCH_NUM):
      bar = tqdm(dataloader, total=len(dataloader))
      for inputs, labels in bar:
          outputs = model(inputs)
          inputs = paddle.round(paddle.reshape(inputs*config.NORMALIZED, shape=(-1,)))
          outputs = paddle.round(paddle.reshape(outputs*config.NORMALIZED, shape=(-1,)))
          print(inputs.tolist())
          print(outputs.tolist())

if __name__ == "__main__":
    config = EasyDict()
    # 训练配置
    config.EPOCH_NUM = 1
    config.BATCH_SIZE = 1
    config.SHUFFLE = True
    config.DROP_LAST = True
    config.LEARNING_RATE = 1e-2
    config.DROPOUT_RATE = None
    # 模型配置
    config.MODEL_NAME = "sale_regressor64_100"
    config.SAVE_EPOCH = 100
    config.NUM_LAYERS = 15
    config.HIDDEN_SIZE = 32
    config.NORMALIZED = 64

    # 加载数据集
    dataloader = DataLoader(SaleDataset(1,config.NORMALIZED),batch_size=config.BATCH_SIZE,shuffle=config.SHUFFLE,drop_last=config.DROP_LAST)
    # 加载模型
    model = SaleModel(config.HIDDEN_SIZE, num_layers=config.NUM_LAYERS, dropout_rate=config.DROPOUT_RATE)

    # 保存模型，包含两部分：模型参数和优化器参数
    param_dict = paddle.load("logs/{}.pdparams".format(config.MODEL_NAME))
    model.load_dict(param_dict)
    predict(model, dataloader, config)