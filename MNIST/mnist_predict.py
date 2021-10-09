import paddle

import numpy as np
from paddle.fluid.layers import tensor

from mnist import MNIST

mnist = paddle.vision.datasets.MNIST(mode='test')

sample = mnist[167:213]

def norm_img(img):
    assert len(img.shape) == 3
    batch_size, img_h, img_w = img.shape
    # 归一化图像数据
    img = img/255
    img = np.reshape(img, [batch_size, img_h*img_w])
    return img
    
model = MNIST()
params_file_path = "mnist.pdparams"
param_dict = paddle.load(params_file_path)

model.eval()
tensor_img = norm_img(np.expand_dims(sample[0], axis=0))
result = model(paddle.to_tensor(tensor_img))
print('result',result)
print(type(result))
print("本次预测的数字是", result.numpy().astype('int32'))
print("实际结果的数字是", sample[1])