# encoding = utf-8
import paddle
from paddle.hapi import model
from paddle.nn import Linear, Layer
import paddle.nn.functional as F

import numpy as np
from tqdm import tqdm

# 指定加载图片的库，确保加载的数据是ndarray类型
paddle.vision.set_image_backend('cv2')
train_dataset = paddle.vision.datasets.MNIST(mode='train')

train_data = np.array(train_dataset[0][0])
train_label = np.array(train_dataset[0][1])

class MNIST(Layer):
    def __init__(self) -> None:
        super().__init__()
        self.fc = Linear(in_features=784, out_features=1)
    
    def forward(self, inputs):
        outputs = self.fc(inputs)
        return outputs

model = MNIST()

def norm_img(img):
    assert len(img.shape) == 3
    batch_size, img_h, img_w = img.shape
    # 归一化图像数据
    img = img/255
    img = paddle.reshape(img, [batch_size, img_h*img_w])
    return img

def train(model):
    # 启动训练模式
    model.train()
    # 加载训练集
    train_loader = paddle.io.DataLoader(paddle.vision.datasets.MNIST(mode='train'),
                        batch_size=16, shuffle=True)
    # 定义优化器
    opt = paddle.optimizer.SGD(learning_rate=0.001, parameters=model.parameters())
    
    EPOCH_NUM = 10
    bar = tqdm(range(EPOCH_NUM))
    for epoch in bar:
        for batch_id, data in enumerate(train_loader()):
            images = norm_img(data[0]).astype('float32')
            labels = data[1].astype('float32')

            predicts = model(images)

            loss = F.square_error_cost(predicts, labels)
            avg_loss = paddle.mean(loss)

            bar.postfix = "epoch_id: {}, batch_id: {}, loss is: {}".format(epoch, batch_id, avg_loss.numpy())
            
            avg_loss.backward()
            opt.step()
            opt.clear_grad()


if __name__ == "__main__":
    train(model)
    paddle.save(model.state_dict(), './mnist.pdparams')
