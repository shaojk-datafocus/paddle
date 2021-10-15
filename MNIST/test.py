import paddle
import numpy as np
from paddle.framework import dtype
import paddle.nn.functional as F

x = paddle.to_tensor([[1,0,0,0]],dtype='float32')
labels = paddle.to_tensor([[0,1,0,0]],dtype='int64')



# loss = F.softmax_with_cross_entropy(x, labels)
loss = F.cross_entropy(x, labels, use_softmax=False, soft_label=True)
print(loss.numpy())

acc = paddle.metric.accuracy(input=x, label=labels)
print(acc.numpy())
