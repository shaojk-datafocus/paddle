# import paddle
# from paddle.nn import LSTM
# import numpy as np

# # model = LSTM(input_size=1,hidden_size=8,num_layers=4)

# # inputs = paddle.ones((2,10,1),dtype="float32")

# # outputs, (last_hidden, last_cell)  = model(inputs)
# # print(last_hidden)

# # optimizer = paddle.optimizer.Adam(learning_rate=0.1,parameters=model.parameters())
# # print(optimizer.state_dict())

# # loss_map = paddle.to_tensor([i for i in range(10)]).astype("float32")
# # loss_map = paddle.expand(loss_map, shape=(3,10))
# # print(loss_map)
# i = 300
# num_seq = np.linspace(i,i-14,15,dtype="float32")
# print(num_seq)

from datetime import datetime, timedelta

date1 = datetime(2021, 10, 18, 0, 0)
print(date1)
date1+=timedelta(days=1)
print(date1)
l = [1,2,3]
l.extend([9])
l.reverse()
print(l)