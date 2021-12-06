from operator import mod
from re import S, T
import paddle
from paddle.fluid.layers.nn import pad, shape
from paddle.framework import dtype
from paddle.nn import Layer, LSTM, Dropout, Linear, Conv1D

class SaleModel(Layer):
  def __init__(self, input_size, hidden_size, num_layers=1, init_scale=0.1, dropout_rate=None):
    super(SaleModel, self).__init__()
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.dropout_rate = dropout_rate
    self.init_scale = init_scale
    self.conv = Conv1D(in_channels=1, out_channels=self.hidden_size, kernel_size=3, padding=1)
    self.lstm = LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers)
    init_weight = paddle.ParamAttr(initializer=paddle.fluid.initializer.MSRA(uniform=False))
    init_bias = paddle.ParamAttr(initializer=paddle.fluid.initializer.MSRA(uniform=False))
    self.fc = Linear(in_features=self.hidden_size,out_features=1, weight_attr=init_weight,bias_attr=init_bias)
    self.dropout_layer = Dropout(p=self.dropout_rate)
    
  def forward(self, x, **kwargs):
    batch_size = x.shape[0]
    x = self.conv(x)
    # if self.dropout_rate is not None and self.dropout_rate>0:
    #   x = self.dropout_layer(x)

    init_hidden_data = paddle.zeros(
      (self.num_layers, batch_size, self.hidden_size), dtype='float32')
    init_hidden_data.stop_gradient = True
    init_cell_data = paddle.zeros(
      (self.num_layers, batch_size, self.hidden_size), dtype='float32')
    init_cell_data.stop_gradient = True

    x = paddle.transpose(x, perm=[0, 2, 1])
    _, (last_hidden, last_cell) = self.lstm(x, (init_hidden_data, init_cell_data))

    return self.fc(last_hidden[-1])

if __name__ == "__main__":
  inputs = paddle.ones((6,1,2))
  model = SaleModel(1,4,5, num_layers=3)
  outputs = model(inputs)
  print(model.fc.weight.shape)
  print(outputs)