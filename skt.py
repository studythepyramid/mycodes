#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 02:15:51 2025

@author: za
"""
import matplotlib.pyplot as plt
import numpy as np

# X = np.c_[0.5, 1].T
# y = [0.5, 1]
# X_test = np.c_[0, 2].T


# from sklearn import linear_model

# # regr = linear_model.LinearRegression()
# # regr.fit(X, y)
# # plt.plot(X, y, "o")
# # plt.plot(X_test, regr.predict(X_test))
# # plt.show()

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())

def generating_func(x, rng, err=0.5):
    return rng.normal(10 - 1. / (x + 0.1), err)
# randomly sample more data
rng = np.random.default_rng(27446968)
x = rng.random(size=200)
y = generating_func(x, err=1., rng=rng)

# plt.scatter(x,y)
# plt.show()


xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.4)


from sklearn.model_selection import validation_curve
degrees = np.arange(1, 21)
model = make_pipeline(PolynomialFeatures(), LinearRegression())
# Vary the "degrees" on the pipeline step "polynomialfeatures"
train_scores, validation_scores = validation_curve(
                model, x[:, np.newaxis], y,
                param_name='polynomialfeatures__degree',
                param_range=degrees)


# Plot the mean train score and validation score across folds
plt.plot(degrees, validation_scores.mean(axis=1), label='cross-validation')
plt.plot(degrees, train_scores.mean(axis=1), label='training')
plt.legend(loc='best')
plt.show()

