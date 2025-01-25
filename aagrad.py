#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo what's gradient and tensor

Created on Thu Jan  2 16:12:15 2025

@author: za
"""
import torch

# create tensor without requires_grad = true
x = torch.tensor(3)

# create tensors with requires_grad = true
w = torch.tensor(2.0, requires_grad = True)
b = torch.tensor(5.0, requires_grad = True)

# print the tensors
print("x:", x)
print("w:", w)
print("b:", b)

# define a function y for the above tensors
y = w*x + b
print("y:", y)

# Compute gradients by calling backward function for y
y.backward()

# Access and print the gradients w.r.t x, w, and b
dx = x.grad
dw = w.grad
db = b.grad
print("x.grad :", dx)
print("w.grad :", dw)
print("b.grad :", db)


## Another demo

print("Another demo")
# create tensors with requires_grad = true
x = torch.tensor(3.0, requires_grad = True)
y = torch.tensor(4.0, requires_grad = True)

# print the tensors
print("x:", x)
print("y:", y)

# define a function z of above created tensors
z = x**y
print("z:", z)

# call backward function for z to compute the gradients
z.backward()

# Access and print the gradients w.r.t x, and y
dx = x.grad
dy = y.grad
print("x.grad :", dx)
print("y.grad :", dy)


## Demo with weight change to mysterial things
print("Demo with weight change to mysterial things")

# create tensor without requires_grad = true
x = torch.tensor(2)

# create tensors with requires_grad = true
w = torch.tensor(2.0, requires_grad = True)
b = torch.tensor(5.0, requires_grad = True)

# print the tensors
print("x:", x)
print("w:", w)
print("b:", b)

# define a function y for the above tensors
y = w*x + b
print("y:", y)

y_target = torch.tensor(10.0, requires_grad=True)
print("y_target:", y_target)

# loss = (y_target - y)**2
loss = abs(y_target - y)
print("loss:", loss)
# Compute gradients by calling backward function for y
loss.backward()

# Access and print the gradients w.r.t x, w, and b
dx = x.grad
dw = w.grad
db = b.grad
dyt= y_target.grad
print("x.grad :", dx)
print("w.grad :", dw)
print("b.grad :", db)
print("y_target.grad :", dyt)

