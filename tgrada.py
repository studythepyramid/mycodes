#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 18:46:45 2024

@author: za
"""

import torch

a = torch.tensor([2., 3.], requires_grad=True)
b = torch.tensor([6., 4.], requires_grad=True)

Q = 3*a**3 - b**2

# external_grad = torch.tensor([1., 1.])
# Q.backward(gradient=external_grad)

# check if collected gradients are correct
print(9*a**2 == a.grad)
print(-2*b == b.grad)

W = (3 * Q).abs().sum()
W.backward()


## 2nd demo of gradient and tensor

# create tensors with requires_grad = true
x = torch.tensor(2.0, requires_grad = True)

# print the tensor
print("x:", x)

# define a function y for the tensor, x
y = x**2 + 1
print("y:", y)

# Compute gradients using backward function for y
y.backward()

# Access the gradients using x.grad
dx = x.grad
print("x.grad :", dx)