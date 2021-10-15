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
        self.fc = Linear(980,10)
        print("初始化MNIST模型")

    def forward(self, inputs, labels=None):
        x = self.conv1(inputs)
        x = F.relu(x)
        x = self.max_pool(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.max_pool(x)
        x = paddle.reshape(x, [x.shape[0],-1])
        x = self.fc(x)
        # x = F.softmax(x)
        if labels is not None:
            acc = paddle.metric.accuracy(input=x, label=labels)
            return x, acc
        else:
            return x

if __name__ == '__main__':
    model = MNIST()
    model.eval()
    inputs = paddle.ones([1,1,28,28],dtype='float32')
    # param_dict = paddle.load('mnist.pdparams')
    # model.load_dict(param_dict)
    outputs = model(inputs)
    print(outputs.numpy())