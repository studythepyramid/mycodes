#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 18:57:29 2024

@author: za
"""
import torch
nn = torch.nn

rnn = nn.LSTM(10, 20, 2)
tr = torch.randn(5, 3, 10)
h0 = torch.randn(2, 3, 20)
c0 = torch.randn(2, 3, 20)
output, (hn, cn) = rnn(tr, (h0, c0))


loss = nn.MSELoss()
rin = torch.randn(3, 5, requires_grad=True)
target = torch.randn(3, 5)
msel = loss(rin, target)
# print(msel)
msel.backward()


criterion = torch.nn.MSELoss(reduction="mean")
optimizer = torch.optim.Adam(rnn.parameters(), lr=0.01)


a = torch.tensor((3,3), dtype=torch.float32, requires_grad=True)
b = torch.ones((3,3), dtype=torch.float32, requires_grad=True)

print(f"a : {a}, b: {b}")

b * 3
b.grad