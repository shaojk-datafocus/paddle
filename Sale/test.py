from operator import mod
import paddle
from paddle.nn import LSTM
from paddle.tensor.creation import arange


model = LSTM(input_size=1,hidden_size=8,num_layers=4)

# inputs = paddle.ones((2,10,1),dtype="float32")

# outputs, (last_hidden, last_cell)  = model(inputs)
# print(last_hidden)

# optimizer = paddle.optimizer.Adam(learning_rate=0.1,parameters=model.parameters())
# print(optimizer.state_dict())

param_dict = paddle.load("logs/{}.pdopt".format("sale_regressor_100"))
print(param_dict)