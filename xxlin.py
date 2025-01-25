#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 11:11:20 2025

@author: za
"""

# for f(x) = 2 x^2 + 1
import torch
from torch import nn
import numpy as np

# Linear Simple Module
class LSModel(torch.nn.Module):
    def __init__(self) -> None:
        super(LSModel,self).__init__()
        self.linearLayer = torch.nn.Linear(1,1)
        self.n5 = nn.Sequential(
            nn.Linear(1, 3),
            nn.ReLU(),
            nn.Linear(3,3),
            nn.ReLU(),
            nn.Linear(3, 1)
        )


    def forward(self,x):
        # pred = self.linearLayer(x)
        pred = self.n5(x)
        return pred


model = LSModel()
crit = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(),lr = 0.01)

# Assuming x_data and y_label are tensors with shape (num_samples, 1)
# Replace with your actual data loading or generation logic
# Example:
num_samples = 100
x_data = torch.randn(num_samples, 1) # Generate random input data
y_label = 2 * x_data * x_data + 1 # Generate corresponding labels (example linear relationship)


#main train loop
for epoch in range(1000):

    y_pred = model(x_data)
    loss = crit(y_pred,y_label)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch+1) % 100 == 0:
      print(f"Epoch {epoch+1}, Loss: {loss.item()}")
      
      
x_data.shape
y_pred.shape
import matplotlib.pyplot as plt

plt.scatter(x_data, y_label, color='brown', marker='o', label='Actual')

plt.scatter(x_data, y_pred.detach().numpy(), marker='+', label='Predicted')
plt.legend()
plt.show()