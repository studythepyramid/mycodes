# -*- coding: utf-8 -*-
"""again.nn.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1OqkUF037KFPbqkAXqqEYchjaklnU5QvD

2025 Jan 31
Learn nn.RNN, 
"""

import torch
import torch.nn as nn
import numpy as np

from torch.utils.data import DataLoader, TensorDataset


import matplotlib.pyplot as plt


# from google.colab import drive
# drive.mount('/content/gdrive')
#
# !mkdir /tmp/gdrive
# drive.mount('/tmp/gdrive')
# %cd /content/gdrive/My Drive/Colab Notebooks/

import decimal

#big256bits = decimal.Decimal(2**256)
#print(f"{big256bits:.4E}")

# rnn = nn.RNN(10, 20, 2)
# input = torch.randn(5, 3, 10)
# h0 = torch.randn(2, 3, 20)
# output, hn = rnn(input, h0)

input_dim = 1
hidden_dim = 20 # small?
num_layers = 2
output_dim = 1

learning_rate = 0.01
num_epochs = 100

# rnn = nn.RNN(input_dim, hidden_dim, num_layers, batch_first=True)

# nn.Module?
class RnnModel(nn.Module):
  def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
    super(RnnModel, self).__init__()
    # self.hidden_dim = hidden_dim
    # self.num_layers = num_layers

    self.NLH = None
    self.hidden = None
    self.rnn = nn.RNN(
        input_dim, hidden_dim, num_layers, batch_first=True)
    self.fc = nn.Linear(hidden_dim, output_dim) #X
    self.lrelu = nn.LeakyReLU(0.5)


  def forward(self, x):
    x = x[:,None,None]
    if self.NLH == None: #?
      self.NLH = x.shape

    if self.hidden == None:
      self.hidden = torch.zeros(num_layers, x.size(0), hidden_dim
                     ).requires_grad_()

    out, self.hidden = self.rnn(x, self.hidden.detach())
    out = self.lrelu(out)

    self.hidden = self.lrelu(self.hidden)

    #out = self.fc(out[:, -1, :])
    #out = self.lrelu ?
    return out

  def init_hidden(self):
    self.hidden = None;

oneRnn = RnnModel(input_dim, hidden_dim, num_layers, output_dim)
xi = torch.randn(5, 1, 1)


#"""# data loader and training?"""

x = np.linspace(-20.0, 20.0, 100, dtype=np.float32)
y = np.sin(x/3.14)
# print(x.shape)
# print(y.shape)

ttx = torch.tensor(x, dtype=torch.float32)
tty = torch.tensor(y, dtype=torch.float32)
dsxy = TensorDataset(ttx, tty)
dlxy = DataLoader(dsxy, batch_size=32, shuffle=True)
# train_ds = TensorDataset(x, y)
# train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)

#plt.scatter(x,y)
#plt.show()

#x5 = x[:5]

#x5.view(-1, 1, 1).size(0)



# y = oneRnn(xi)
# print(y)
# 
# import torch.nn.functional as F
# import torch.optim as optim
# 
# loss_func = F.cross_entropy
# 
# # model = oneRnn
# # print(loss_func(model(xb), yb))
# 
# def get_model(lr):
#     model = RnnModel(input_dim, hidden_dim, num_layers, output_dim)
#     return model, optim.SGD(model.parameters(), lr=lr)
# 
# model, opt = get_model(learning_rate)
# 





# class Lambda(nn.Module):
#     def __init__(self, func):
#         super().__init__()
#         self.func = func
# 
#     def forward(self, x):
#         return self.func(x)
# 
# 
# def preprocess(x):
#     return x.view(-1, 1, 1) #?
# 
# def shape_data(x):
#     return x.view(-1, 1, 1) #?
# 
# #?
# model = nn.Sequential(
#     Lambda(preprocess),
#     nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
#     nn.ReLU(),
#     nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
#     nn.ReLU(),
#     nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1),
#     nn.ReLU(),
#     nn.AvgPool2d(4),
#     Lambda(lambda x: x.view(x.size(0), -1)),
# )
# 
# opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
# 
# """# Yahoo Finance data"""
# 
# import yfinance as yf
# 
# # Define the ticker symbol
# #ticker = 'AAPL'
# ticker = 'BTC-USD'
# 
# # Get historical market data
# data = yf.download(ticker, start='2017-06-01', end='2024-12-31')
# 
# # Display the first few rows of the data
# # print(data.head())
# 
# dlit = iter(train_dl)
# xi, yi = next(dlit)
# print(xi.shape)
# print(yi.shape)
# 
# for epoch in range(2):
#     for xb, yb in train_dl:
#         pred = model(xb)
#         loss = loss_func(pred, yb)
# 
#         loss.backward()
#         opt.step()
#         opt.zero_grad()
# 
# print(loss_func(model(xb), yb))
# 
# print(loss_func(model(xb), yb))
# 
# a = torch.tensor([1.1, 2.2], requires_grad=True)
# b = torch.tensor([3.3, 2.2], requires_grad=True)
# # loss_func(a, b)
# 
# x5 = xt[:5, None, None]
# print(x5)
# print(x5.shape)
# p5 = model(x5)
# # print(p10)
# loss_func(p5.squeeze(), yt[:5])
# # p2 = p10.squeeze()
# # print(p2)
# # loss_func(p2, y10)
# # p2 - y10
# 
# xin = xt[:,None,None]
# pred = model(xin)
# 
# loss = loss_func(pred.squeeze(), yt)
# 
# loss.backward()
# opt.step()
# opt.zero_grad()
# 
# print(loss)
# 
# """# Data preparation"""
# 
# # Generate synthetic sine wave data
# t = np.linspace(-20.0, 20.0, 500, dtype=np.float32)
# fsin = np.sin(t/3.0)
# # plt.scatter(t, fsin)
# 
# xt, yt = map(torch.tensor, (t, fsin))
# # xt, yt = xt.unsqueeze(1), yt.unsqueeze(1)
# print(xt.shape)
# print(yt.shape)
# 
# x10 = xt[:10]
# y10 = yt[:10]
# xx = x10[:, None, None]
# print(xx, xx.shape)
# 
# 
# 
# """# Methods for variated input length for RNN"""
# 
# import torch
# import torch.nn as nn
# from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
# 
# class VariableLengthRNN(nn.Module):
#     def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
#         super(VariableLengthRNN, self).__init__()
#         self.hidden_dim = hidden_dim
#         self.num_layers = num_layers
#         self.rnn = nn.RNN(input_dim, hidden_dim, num_layers, batch_first=True)
#         self.fc = nn.Linear(hidden_dim, output_dim)
# 
#     def forward(self, x, lengths):
#         # Pad the input sequences
#         x_padded = pad_sequence(x, batch_first=True, padding_value=0)
# 
#         # Pack the padded sequences
#         packed_input = pack_padded_sequence(x_padded, lengths, batch_first=True, enforce_sorted=False)
# 
#         # Pass the packed sequence through the RNN
#         packed_output, hn = self.rnn(packed_input)
# 
#         # Unpack the output (if needed)
#         output, _ = pad_packed_sequence(packed_output, batch_first=True)
# 
#         # Apply the fully connected layer
#         out = self.fc(output[:, -1, :])  # Get the last hidden state
#         return out
# 
# # Example Usage:
# # Assume 'sequences' is a list of variable-length tensors
# sequences = [torch.tensor([1, 2, 3]), torch.tensor([4, 5]), torch.tensor([6])]
# lengths = [len(seq) for seq in sequences]  # Get lengths of sequences
# 
# # Create an instance of the model
# model = VariableLengthRNN(input_dim=1, hidden_dim=10, num_layers=1, output_dim=1)
# 
# # Forward pass
# output = model(sequences, lengths)
# 
# # ... (rest of your training loop)
# 
# """# Mis LSTM Model"""
# 
# ## https://medium.com/swlh/stock-price-prediction-with-pytorch-37f52ae84632
# 
# class LSTM(nn.Module):
#     def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
#         super(LSTM, self).__init__()
#         self.hidden_dim = hidden_dim
#         self.num_layers = num_layers
# 
#         self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
#         self.fc = nn.Linear(hidden_dim, output_dim)
# 
#     def forward(self, x):
#         h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
#         c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
#         out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
#         out = self.fc(out[:, -1, :])
#         return out



