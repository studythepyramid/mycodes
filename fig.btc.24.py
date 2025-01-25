#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  1 09:55:36 2025

@author: za
"""


import dfread


import numpy as np

#from torch.optim import lr_scheduler


# read in Bitcoin price csv file to pandas Dataframe, 
# with price and volume data during in 2024 Q1
btcdf = dfread.read_btcusd_2024_q1()

btcdf.columns = ["timestamp", "priceUsd", "coinsTraded"]


## plot the outline, 2 hour a price 
twoHr = btcdf[::7200]

import pandas as pd
twoHr = twoHr.copy() # leave source untouched
twoHr['date'] = pd.to_datetime(twoHr['timestamp'], unit='s')

twoHr.plot.scatter(x='date', y='priceUsd', rot=45, xlabel='2hr intervals')
# plt.show()

# third = twoHr.iloc[380:]
# # third.plot.line(x='date', y='priceUsd')
# third.plot.scatter(x='date', y='priceUsd')

plt.show()