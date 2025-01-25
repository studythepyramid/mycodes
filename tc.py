#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 13:17:35 2024

@author: za
"""

import torch

src = torch.arange(1, 11).reshape((2, 5))
print(src)

index = torch.tensor([[1, 1, 2, 0, 2]])
print("#####")

m35 = torch.zeros(3, 5, dtype=src.dtype).scatter_(0, index, src)
# print(m35)


index = torch.tensor([[0, 1, 2], [0, 1, 0]])
d125 = torch.zeros(3, 5, dtype=src.dtype).scatter_(0, index, src)


index = torch.tensor([[0, 1, 2], [0, 1, 4]])
d135 = torch.zeros(3, 5, dtype=src.dtype).scatter_(1, index, src)

m2 = torch.full((2, 4), 2.).scatter_(1, torch.tensor([[2], [3]]),
           1.23, reduce='multiply')
m4 = torch.full((2, 4), 2.).scatter_(1, torch.tensor([[2], [3]]),
           1.23, reduce='add')

ltt = torch.zeros(    10, 
                  dtype=torch.float).scatter_(dim=0, index=torch.tensor(1) )