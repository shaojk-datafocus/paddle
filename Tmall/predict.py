import paddle
from paddle.fluid.layers.nn import pad
from paddle.io import DataLoader
from model import TmallModel
from dataset import TmallDataset
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
  config.NORMALIZED = paddle.to_tensor(config.NORMALIZED).astype("float32")
  i = 0
  for epoch in range(config.EVAL_EPOCH_NUM):
      bar = tqdm(dataloader, total=len(dataloader))
      for inputs, labels, _ in bar:
          outputs = model(paddle.transpose(inputs, perm=(0,2,1)))
          inputs = paddle.round(inputs[0]*config.NORMALIZED)
          labels = paddle.round(paddle.reshape(labels*config.NORMALIZED[0], shape=(-1,)))
          outputs = paddle.round(paddle.reshape(outputs*config.NORMALIZED[0], shape=(-1,)))
          if labels[-1] > 0:
            print(inputs.tolist())
            print(labels.tolist())
            print(outputs.tolist())
            i+=1
          if i>5:
            return 

if __name__ == "__main__":
    from config import config

    # 加载数据集
    dataloader = DataLoader(TmallDataset(config.LEN_SEQ, config.UNKNOWN_SEQ, config.NORMALIZED),batch_size=config.EVAL_BATCH_SIZE,shuffle=config.SHUFFLE,drop_last=config.DROP_LAST)
    # 加载模型
    model = TmallModel(config.INPUT_SIZE, config.HIDDEN_SIZE, num_layers=config.NUM_LAYERS, dropout_rate=config.DROPOUT_RATE)
    # 保存模型，包含两部分：模型参数和优化器参数
    param_dict = paddle.load("{}/{}.pdparams".format(config.LOG_PATH,config.MODEL_NAME))
    model.load_dict(param_dict)
    predict(model, dataloader, config)