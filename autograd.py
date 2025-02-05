#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 14:31:42 2024

@author: za
"""

import torch
from torchvision.models import resnet18, ResNet18_Weights
model = resnet18(weights=ResNet18_Weights.DEFAULT)

data = torch.rand(1, 3, 64, 64)
labels = torch.rand(1, 1000)

pred = model(data)
loss = (pred - labels).sum()
lb = loss.backward()

optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
optim.step()

