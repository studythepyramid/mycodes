#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 18:16:56 2025

@author: za
"""

import numpy as np
x = np.arange(9).reshape((3,3))
y = np.arange(3)

print( np.dot(x,y))


# def guard_zero(operate):
#     def inner(x, y):
#         if y == 0:
#             print("Cannot divide by 0.")
#             return
#         return operate(x, y)
#     return inner


# @guard_zero
# def divide(x, y):
#     return x / y

# print(divide(5,0))

# print(divide(5,3))