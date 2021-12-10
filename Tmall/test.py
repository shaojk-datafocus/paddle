# import random
# import paddle
# from paddle.nn import LSTM
# import numpy as np
# from datetime import datetime, timedelta

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

# a = np.array([1,0,1,0,2,3,5])
# b = a>=2
# c = a==1
# print(np.bitwise_or(b,c))

a = [14,6,9,4,8,1,6]
b = [14,6,9,4,89,1,6]

# f = list(filter(lambda x: x>7, a))
# print(a.index(random.choice(f)))

print(a==b)