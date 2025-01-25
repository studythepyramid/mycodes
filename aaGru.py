#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  1 23:32:54 2025

@author: za
"""

import torch
from torch import nn

inputSize = 2
hiddenSize = 2
outputSize = 2
gru = nn.GRU(inputSize, hiddenSize, outputSize, batch_first=True)
inputt = torch.randn(1,1,2)
h0 = torch.randn(2,1,2)

out, hn = gru(inputt, h0)


# Assuming your data is a list of (time, price) tuples:
data = [(1677340800, 23100), (1677427200, 23250), ...]  # Example data

# Convert time to numerical representation (e.g., Unix timestamps):
data = [(time, price) for time, price in data]  

# Convert data to tensor:
data_tensor = torch.tensor(data, dtype=torch.float32)

# ... (Preprocessing, train/test split, etc.) ...

# Instantiate the model with input_size = 2
input_size = 2  # Now considering both time and price
hidden_size = 50
output_size = 1
model = LSTMModel(input_size, hidden_size, output_size)

# ... (Training loop) ...


# inputt = torch.randn(2,1,1)
# h0 = torch.randn(1, 2, 1)
# gru = nn.GRU(1, 1, 1, batch_first=True)
# out, hn = gru(inputt, h0)
# gru.parameters()




# rnn = nn.GRU(10, 20, 2)
# input = torch.randn(5, 3, 10)
# h0 = torch.randn(2, 3, 20)
# output, hn = rnn(input, h0)

# print(output.shape, hn.shape)
# mp = rnn.parameters()

# criterion = nn.MSELoss()
# optimizer = torch.optim.Adam(rnn.parameters(), lr=0.01)

# # # Compute loss
# # loss = criterion(outputs, trainY)
# # loss.backward()
# # optimizer.step()

