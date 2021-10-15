import paddle
import paddle.nn.functional as F

import numpy as np
from mnist import MNIST

mnist = paddle.vision.datasets.MNIST(mode='test')

samples = (134,163,164,186,268,467)
images = np.array([mnist[i][0] for i in samples])
labels = np.array([mnist[i][1] for i in samples])

def norm_img(img):
    assert len(img.shape) == 3
    batch_size, img_h, img_w = img.shape
    # 归一化图像数据
    img = img/255
    img = np.reshape(img, [batch_size, 1,img_h, img_w])
    return img
    
model = MNIST()
# 加载模型
param_dict = paddle.load("mnist.pdparams")
model.load_dict(param_dict)
# 设置预测模式
model.eval()
# 数据预处理
tensor_img = norm_img(images)
inputs = paddle.to_tensor(tensor_img)

result = model(inputs)
result = F.softmax(result)

print("本次预测的数字是", np.argmax(result.numpy(),axis=1))
print("实际结果的数字是", labels)