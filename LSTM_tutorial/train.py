import paddle
from paddle.nn import functional as F
from paddle.io import DataLoader
from easydict import EasyDict
from model import SaleModel
from dataset import SaleDataset
from tqdm import tqdm

def train(model, dataloader, config, logger=None):
  # 检测是否可以使用GPU，如果可以优先使用GPU
  use_gpu = True if paddle.get_device().startswith("gpu") else False
  if use_gpu:
      paddle.set_device('gpu:0')

  # 指定优化策略，更新模型参数
  optimizer = paddle.optimizer.Adam(learning_rate=config.LEARNING_RATE, beta1=0.9, beta2=0.999, parameters=model.parameters()) 
  # 加载模型权重
  if os.path.exists("logs/{}.pdparams".format(config.MODEL_NAME)):
      param_dict = paddle.load("logs/{}.pdparams".format(config.MODEL_NAME))
      model.load_dict(param_dict)
      print("加载已保存模型权重","logs/{}.pdparams".format(config.MODEL_NAME))
  if os.path.exists("logs/{}.pdopt".format(config.MODEL_NAME)):
      param_dict = paddle.load("logs/{}.pdopt".format(config.MODEL_NAME))
      optimizer.set_state_dict(param_dict)
      print("加载已保存优化器权重","logs/{}.pdopt".format(config.MODEL_NAME))
    

  # 开启模型训练模式
  model.train()
  
  i = 0
  focus_map = paddle.to_tensor([i for i in range(model.num_layers)])/model.num_layers
  focus_map = paddle.expand(focus_map, shape=(config.BATCH_SIZE,config.NUM_LAYERS))
  for epoch in range(config.EPOCH_NUM):
      bar = tqdm(dataloader, total=len(dataloader))
      for inputs,labels in bar:
          outputs = model(inputs)
          outputs = paddle.transpose(outputs, perm=[2,1,0])
          outputs = paddle.reshape(outputs, shape=(-1,model.num_layers))
          loss = F.mse_loss(outputs,labels,reduction='none')
          loss = paddle.sum(loss*focus_map)/config.BATCH_SIZE
          loss.backward()
          optimizer.step()
          optimizer.clear_grad()

          bar.postfix = "{}/{}, Loss: {:.6f} Output:{:.0f},{:.0f}".format(epoch+1, config.EPOCH_NUM, loss.numpy()[0], labels.tolist()[0][-1]*config.NORMALIZED,outputs.tolist()[0][-1]*config.NORMALIZED)
          if logger and i%100==0:
              logger.add_scalar(tag = 'loss', step = i, value = loss.numpy()[0])
          i += 1
      if (epoch+1) % config.SAVE_EPOCH == 0:
          paddle.save(model.state_dict(), "logs/{}.pdparams".format(config.MODEL_NAME))
          paddle.save(optimizer.state_dict(), "logs/{}.pdopt".format(config.MODEL_NAME))
          print("保存训练权重","logs/{}".format(config.MODEL_NAME))

if __name__ == "__main__":
    import os
    config = EasyDict()
    # 训练配置
    config.EPOCH_NUM = 200
    config.BATCH_SIZE = 64
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
    dataloader = DataLoader(SaleDataset(10000,config.NORMALIZED),batch_size=config.BATCH_SIZE,shuffle=config.SHUFFLE,drop_last=config.DROP_LAST)
    # 加载模型
    model = SaleModel(config.HIDDEN_SIZE, num_layers=config.NUM_LAYERS, dropout_rate=config.DROPOUT_RATE)
    # 训练
    train(model, dataloader, config)
    