# encoding = utf-8
import paddle
from paddle.fluid.layers.nn import pad
from paddle.nn import weight_norm
import paddle.nn.functional as F

import numpy as np
from easydict import EasyDict
from tqdm import tqdm
from model import MNIST


def data_preprocessing(data):
    num_class = 10
    batch_size, img_h, img_w = data[0].shape
    # 处理图片
    images = paddle.reshape(data[0]/255, [batch_size, 1, img_h, img_w]).astype('float32')
    # 制作标签
    labels = paddle.reshape(data[1],[batch_size, 1]).astype('int64')
    # labels = np.zeros((batch_size, num_class))
    # for i, index in enumerate(data[1].numpy()):
    #     labels[i][index[0]] = 1
    # labels = paddle.to_tensor(labels).astype('int64')
    return images, labels

def train(model, config, logger=None):
    # GPU配置
    paddle.set_device('gpu:0') if config.USE_GPU else paddle.set_device('cpu')
    # 启动训练模式
    model.train()
    # 加载训练集
    paddle.vision.set_image_backend('cv2')
    train_dataset = paddle.vision.datasets.MNIST(mode='train')
    train_loader = paddle.io.DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    

    # 定义优化器
    opt = paddle.optimizer.Adagrad(
        learning_rate=config.LEARNING_RATE, 
        weight_decay=paddle.regularizer.L2Decay(coeff=config.COEFF),
        parameters=model.parameters())

    i = 0
    for epoch in range(config.EPOCH_NUM):
        bar = tqdm(train_loader())
        for data in bar:
            images, labels = data_preprocessing(data)

            predicts, acc = model(images, labels)

            loss = F.cross_entropy(predicts, labels)
            avg_loss = paddle.mean(loss)

            bar.postfix = "{}/{}, Loss: {:.6f}, Acc: {:.2%}".format(epoch+1, config.EPOCH_NUM, avg_loss.numpy()[0], acc.numpy()[0])
            
            avg_loss.backward()
            opt.step()
            opt.clear_grad()

            if logger and i%100==0:
                logger.add_scalar(tag = 'acc', step = i, value = acc.numpy())
                logger.add_scalar(tag = 'loss', step = i, value = avg_loss.numpy())
            i += 1

if __name__ == "__main__":
    # from visualdl import LogWriter
    # log_writer = LogWriter("./log")
    model = MNIST()
    config = EasyDict({
        "EPOCH_NUM": 2,
        "BATCH_SIZE": 4,
        "LEARNING_RATE": 0.001,
        "USE_GPU": False,
        "COEFF": 1e-5
    })
    train(model, config)
    paddle.save(model.state_dict(), './mnist.pdparams')
