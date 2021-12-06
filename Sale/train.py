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

  # 开启模型训练模式
  model.train()
  
  i = 0
  for epoch in range(config.EPOCH_NUM):
      bar = tqdm(dataloader, total=len(dataloader))
      for inputs, labels in bar:
          outputs = model(inputs)

          loss = F.mse_loss(outputs,labels)

          loss.backward()
          optimizer.step()
          optimizer.clear_grad()
          
          bar.postfix = "{}/{}, Loss: {:.6f} Output:{:.0f},{:.0f}".format(epoch+1, config.EPOCH_NUM, loss.numpy()[0], labels.tolist()[0],outputs.tolist()[0][0])
          if logger and i%100==0:
              logger.add_scalar(tag = 'loss', step = i, value = loss.numpy()[0])
          i += 1

if __name__ == "__main__":
    config = EasyDict()
    # 训练配置
    config.EPOCH_NUM = 10
    config.BATCH_SIZE = 32
    config.SHUFFLE = True
    config.DROP_LAST = True
    config.LEARNING_RATE = 1e-3
    config.DROPOUT_RATE = None
    # 模型配置
    config.NUM_LAYERS = 8
    config.INPUT_SIZE = 14
    config.HIDDEN_SIZE = 32

    # 加载数据集
    dataloader = DataLoader(SaleDataset(10000),batch_size=config.BATCH_SIZE,shuffle=config.SHUFFLE,drop_last=config.DROP_LAST)
    # 加载模型
    model = SaleModel(config.INPUT_SIZE, config.HIDDEN_SIZE, num_layers=config.NUM_LAYERS, dropout_rate=config.DROPOUT_RATE)
    # 训练
    train(model, dataloader, config)

    # 保存模型，包含两部分：模型参数和优化器参数
    model_name = "sale_regressor"
    # 保存训练好的模型参数
    paddle.save(model.state_dict(), "logs/{}.pdparams".format(model_name))
    # # 保存优化器参数，方便后续模型继续训练
    paddle.save(model.state_dict(), "logs/{}.pdopt".format(model_name))