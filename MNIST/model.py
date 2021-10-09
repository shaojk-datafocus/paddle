import paddle
from paddle.nn import Layer, Conv2D, MaxPool2D, Linear
import paddle.nn.functional as F

class MNIST(Layer):
    def __init__(self, name_scope=None, dtype="float32"):
        super().__init__(name_scope=name_scope, dtype=dtype)
    
        # in, out, kernel, stride, padding
        self.conv1 = Conv2D(1, 20, 5, 1, 2)
        self.conv2 = Conv2D(20,20,5,1,2)
        self.max_pool = MaxPool2D(2,2)
        self.fc = Linear(980,1)
        print("初始化MNIST模型")

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = F.relu(x)
        x = self.max_pool(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.max_pool(x)
        x = paddle.reshape(x, [x.shape[0],-1])
        x = self.fc(x)
        return x