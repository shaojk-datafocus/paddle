# encoding = utf-8
import paddle
import paddle.nn.functional as F

from easydict import EasyDict
from tqdm import tqdm
from model import MNIST

def norm_img(img):
    assert len(img.shape) == 3
    batch_size, img_h, img_w = img.shape
    # 归一化图像数据
    img = img/255
    img = paddle.reshape(img, [batch_size, 1, img_h, img_w])
    return img

def data_preprocessing(data):
    images = norm_img(data[0]).astype('float32')
    labels = data[1].astype('float32')
    return images, labels

def train(model, config):
    # 启动训练模式
    model.train()
    # 加载训练集
    paddle.vision.set_image_backend('cv2')
    train_dataset = paddle.vision.datasets.MNIST(mode='train')
    train_loader = paddle.io.DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)

    # 定义优化器
    opt = paddle.optimizer.Adam(learning_rate=config.LEARNING_RATE, parameters=model.parameters())
    
    
    for epoch in range(config.EPOCH_NUM):
        bar = tqdm(train_loader())
        for data in bar:
            images, labels = data_preprocessing(data)

            predicts = model(images)

            loss = F.square_error_cost(predicts, labels)
            avg_loss = paddle.mean(loss)

            bar.postfix = "{}/{}, loss is: {:.6f}".format(epoch+1, config.EPOCH_NUM, avg_loss.numpy()[0])
            
            avg_loss.backward()
            opt.step()
            opt.clear_grad()


if __name__ == "__main__":
    model = MNIST()
    config = EasyDict({
        "EPOCH_NUM": 10,
        "BATCH_SIZE": 4,
        "LEARNING_RATE": 0.001
    })
    train(model, config)
    paddle.save(model.state_dict(), './mnist.pdparams')
