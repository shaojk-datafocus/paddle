from easydict import EasyDict

config = EasyDict()
# 模型配置
config.MODEL_NAME = "regressor32"
config.NUM_LAYERS = 7
config.INPUT_SIZE = 5
config.HIDDEN_SIZE = 32
# 数据集配置
config.LEN_SEQ = config.NUM_LAYERS
config.UNKNOWN_SEQ = 1
config.NORMALIZED = [5,5,300,6,7]
# 训练配置
config.TRIAN_EPOCH_NUM = 200
config.TRAIN_BATCH_SIZE = 32
config.SAVE_EPOCH = 50
config.START_I = 7200
config.SHUFFLE = True
config.DROP_LAST = True
config.LEARNING_RATE = 1e-2
config.DROPOUT_RATE = None
config.LOG_PATH = "./logs/%s"%config.MODEL_NAME
# 预测配置
config.EVAL_BATCH_SIZE = 16
