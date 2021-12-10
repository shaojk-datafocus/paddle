import paddle
from paddle.nn import functional as F
from paddle.io import DataLoader
from model import TmallModel
from dataset import TmallDataset
from tqdm import tqdm

def train(model, dataloader, config, logger=None):
  # 检测是否可以使用GPU，如果可以优先使用GPU
  use_gpu = True if paddle.get_device().startswith("gpu") else False
  if use_gpu:
      paddle.set_device('gpu:0')

  # 指定优化策略，更新模型参数
  optimizer = paddle.optimizer.Adam(learning_rate=config.LEARNING_RATE, beta1=0.9, beta2=0.999, parameters=model.parameters()) 
  # 加载模型权重
  if os.path.exists("{}/{}.pdparams".format(config.LOG_PATH, config.MODEL_NAME)):
      param_dict = paddle.load("{}/{}.pdparams".format(config.LOG_PATH, config.MODEL_NAME))
      model.load_dict(param_dict)
      print("加载已保存模型权重","{}/{}.pdparams".format(config.LOG_PATH, config.MODEL_NAME))
  if os.path.exists("{}/{}.pdopt".format(config.LOG_PATH, config.MODEL_NAME)):
      param_dict = paddle.load("{}/{}.pdopt".format(config.LOG_PATH, config.MODEL_NAME))
      optimizer.set_state_dict(param_dict)
      print("加载已保存优化器权重","{}/{}.pdopt".format(config.LOG_PATH, config.MODEL_NAME))

  # 开启模型训练模式
  model.train()
  i = config.START_I
  focus_map = paddle.to_tensor([i for i in range(config.LEN_SEQ-config.UNKNOWN_SEQ)]+[1]*config.UNKNOWN_SEQ)/model.num_layers
  focus_map = paddle.expand(focus_map, shape=(config.TRAIN_BATCH_SIZE,config.NUM_LAYERS))
  for epoch in range(config.TRIAN_EPOCH_NUM):
      bar = tqdm(dataloader, total=len(dataloader))
      for inputs,labels,focus in bar:
          outputs = model(paddle.transpose(inputs, perm=(0,2,1)))
          outputs = paddle.transpose(outputs, perm=[2,1,0])
          outputs = paddle.reshape(outputs, shape=(-1,model.num_layers))
          loss = F.mse_loss(outputs,labels,reduction='none')
          focus = paddle.expand(focus, shape=(config.TRAIN_BATCH_SIZE,config.NUM_LAYERS))
          loss = paddle.sum(loss*focus_map*focus)/config.TRAIN_BATCH_SIZE
          loss.backward()
          optimizer.step()
          optimizer.clear_grad()

          bar.postfix = "{}/{}, Loss: {:.6f}".format(epoch+1, config.TRIAN_EPOCH_NUM, loss.numpy()[0])
          if logger and i%10==0:
              logger.add_scalar(tag = 'loss', step = i, value = loss.numpy()[0])
          i += 1
      if (epoch+1) % config.SAVE_EPOCH == 0:
          paddle.save(model.state_dict(), "{}/{}.pdparams".format(config.LOG_PATH,config.MODEL_NAME))
          paddle.save(optimizer.state_dict(), "{}/{}.pdopt".format(config.LOG_PATH,config.MODEL_NAME))
          print("保存训练权重","{}/{}".format(config.LOG_PATH,config.MODEL_NAME))

if __name__ == "__main__":
    import os
    from visualdl import LogWriter
    from config import config

    # 加载数据集
    dataloader = DataLoader(TmallDataset(config.LEN_SEQ, config.UNKNOWN_SEQ, config.NORMALIZED),batch_size=config.TRAIN_BATCH_SIZE,shuffle=config.SHUFFLE,drop_last=config.DROP_LAST)
    # 加载模型
    model = TmallModel(config.INPUT_SIZE, config.HIDDEN_SIZE, num_layers=config.NUM_LAYERS, dropout_rate=config.DROPOUT_RATE)
    # 日志
    logger = LogWriter(logdir=config.LOG_PATH)
    # 训练
    train(model, dataloader, config, logger)
    