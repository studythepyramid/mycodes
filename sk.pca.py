#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 16:22:07 2025

@author: za
"""

from sklearn import datasets

iris = datasets.load_iris()
X = iris.data
y = iris.target

from sklearn.decomposition import PCA

pca = PCA(n_components=2, whiten=True)
# pca.fit(X)

X_pca = pca.transform(X)

target_ids = range(len(iris.target_names))

import matplotlib.pyplot as plt

plt.figure(figsize=(6, 5))
for i, c, label in zip(target_ids, "rgbcmykw", iris.target_names, strict=False):
    # print(f"i {i}, y {y}, {X_pca[y == i, 0], X_pca[y == i, 1]}")
    plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], c=c, label=label)
plt.legend()
plt.show()