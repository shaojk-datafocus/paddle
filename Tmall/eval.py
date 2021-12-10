import paddle
from paddle.io import DataLoader
from model import TmallModel
from dataset import TmallDataset
from tqdm import tqdm
import numpy as np

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
  score = [0] * config.UNKNOWN_SEQ
  outputs_set = []
  labels_set = []
  num_total = 0
  for inputs, labels, _ in tqdm(dataloader, postfix="评估中"):
      outputs = model(paddle.transpose(inputs, perm=(0,2,1)))
      outputs = paddle.transpose(outputs, perm=[2,1,0])
      outputs = paddle.reshape(outputs, shape=(-1,model.num_layers))
      outputs*=config.NORMALIZED[0]
      labels*=config.NORMALIZED[0]
      outputs_set+=paddle.round(outputs)[:,-config.UNKNOWN_SEQ:].tolist()
      labels_set+=labels[:,-config.UNKNOWN_SEQ:].tolist()
      bias = (paddle.abs(outputs-labels)<0.5).astype("int32")
      bias = paddle.sum(bias, axis=0)
      for i,bias in enumerate(bias.tolist()[-config.UNKNOWN_SEQ:]):
        score[i] += bias
      num_total += config.EVAL_BATCH_SIZE
  print("完全准确率")
  for s in score:
    print("%.2f%%"%(s/num_total*100), end=' ')
  outputs_set = np.array(outputs_set)
  labels_set = np.array(labels_set)
  avg_out = np.mean(outputs_set)
  avg_label = np.mean(labels_set)
  r2 = 1-np.sum(np.square(outputs_set-avg_out))/np.sum(np.square(labels_set-avg_label))
  print("R2 Score")
  print(r2)
  # print(r2_score(outputs_set, labels_set))

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