import paddle
from paddle.fluid.layers.nn import pad
from paddle.nn import Layer, LSTM, Dropout, Linear, Conv1D

class SaleModel(Layer):
  def __init__(self, hidden_size, num_layers=1, init_scale=0.1, dropout_rate=None):
    super(SaleModel, self).__init__()
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.dropout_rate = dropout_rate
    self.init_scale = init_scale
    self.conv = Conv1D(in_channels=1, out_channels=self.hidden_size, kernel_size=3, padding=1)
    self.lstm = LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size, num_layers=num_layers)
    self.conv_ = Conv1D(in_channels=self.hidden_size, out_channels=1, kernel_size=3, padding=1)
    # self.fc = Linear(in_features=self.hidden_size,out_features=1)
    # self.dropout_layer = Dropout(p=self.dropout_rate)
    
  def forward(self, x, **kwargs):
    batch_size = x.shape[0]
    # X.shape = batch_size, channel, length
    x = self.conv(x)
    # if self.dropout_rate is not None and self.dropout_rate>0:
    #   x = self.dropout_layer(x)
    init_hidden_data = paddle.zeros((self.num_layers, batch_size, self.hidden_size), dtype='float32')
    init_hidden_data.stop_gradient = True
    init_cell_data = paddle.zeros((self.num_layers, batch_size, self.hidden_size), dtype='float32')
    init_cell_data.stop_gradient = True
    # x = paddle.reshape(x, shape=(batch_size, -1, self.hidden_size))
    x = paddle.transpose(x, perm=[0, 2, 1])
    # X' shape shoudl be like [batch_size, time_steps, hidden_size]
    outputs, (last_hidden, last_cell) = self.lstm(x, (init_hidden_data, init_cell_data))
    last_hidden = paddle.transpose(last_hidden, perm=[0,2,1])
    return self.conv_(last_hidden)

if __name__ == "__main__":
  model = SaleModel(8,num_layers=3)
  inputs = paddle.ones((6,1,14))
  outputs = model(inputs)
  print(outputs.shape)