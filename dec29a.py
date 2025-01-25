#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 29 14:23:33 2024

@author: andrew
"""


import dfread

import torch
nn = torch.nn

import numpy as np

#from torch.optim import lr_scheduler


# read in Bitcoin price csv file to pandas Dataframe, 
# with price and volume data during in 2024 Q1
btcdf = dfread.read_btcusd_2024_q1()

btcdf.columns = ["timestamp", "priceUsd", "coinsTraded"]


## plot the outline, 2 hour a price 
twoHr = btcdf[::7200]

# import pandas as pd
# twoHr = twoHr.copy() # leave source untouched
# twoHr['date'] = pd.to_datetime(twoHr['timestamp'], unit='s')
# twoHr.plot.scatter(x='date', y='priceUsd', rot=45, xlabel='2hr intervals')
# plt.show()


seq_length = 10
def make_data_sequences(data, seq_length):
    xs = []
    ys = []
    for i in range(len(data)-seq_length):
        x = data[i:(i+seq_length)]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# p1k = btcdf['priceUsd'][:1000]
priceTwo = twoHr['priceUsd'].to_numpy()
X, y = make_data_sequences(priceTwo, seq_length)

# Convert data to PyTorch tensors
trainX = torch.tensor(X[:, :, None], dtype=torch.float32)
trainY = torch.tensor(y[:, None], dtype=torch.float32)


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, h0=None, c0=None):
        # If hidden and cell states are not provided, 
        # initialize them as zeros
        if h0 is None or c0 is None:
            h0 = torch.zeros(
                self.layer_dim, x.size(0), self.hidden_dim
                ).to(x.device)
            c0 = torch.zeros(
                self.layer_dim, x.size(0), self.hidden_dim
                ).to(x.device)
        
        # Forward pass through LSTM
        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # Selecting the last output
        return out, hn, cn
    
    
# Initialize model, loss, and optimizer
model = LSTMModel(input_dim=1, hidden_dim=100, layer_dim=1, output_dim=1)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


# Training loop
num_epochs = 100
h0, c0 = None, None  # Initialize hidden and cell states


### steppings

# model.train()
# optimizer.zero_grad()

# # Forward pass
# outputs, h0, c0 = model(trainX, h0, c0)

# # Compute loss
# loss = criterion(outputs, trainY)
# loss.backward()
# optimizer.step()

# # Detach hidden and cell states to 
# # prevent backpropagation through the entire sequence
# h0 = h0.detach()
# c0 = c0.detach()

# loss.item()
      

# for epoch in range(num_epochs):
#     # model.train()
#     optimizer.zero_grad()

#     # Forward pass
#     outputs, h0, c0 = model(trainX, h0, c0)

#     # Compute loss
#     loss = criterion(outputs, trainY)
#     loss.backward()
#     optimizer.step()

#     # Detach hidden and cell states to 
#     # prevent backpropagation through the entire sequence
#     h0 = h0.detach()
#     c0 = c0.detach()

#     if (epoch+1) % 10 == 0:
#         print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        
        
# # Predicted outputs
# model.eval()
# predicted, _, _ = model(trainX, h0, c0)

# # Adjusting the original data and prediction for plotting
# original = data[seq_length:]  # Original data from the end of the first sequence
# time_steps = np.arange(seq_length, len(data))  # Corresponding time steps

# # Plotting
# plt.figure(figsize=(12, 6))
# plt.plot(time_steps, original, label='Original Data')
# plt.plot(time_steps, predicted.detach().numpy(), label='Predicted Data', linestyle='--')
# plt.title('LSTM Model Predictions vs. Original Data')
# plt.xlabel('Time Step')
# plt.ylabel('Value')
# plt.legend()
# plt.show()        


        