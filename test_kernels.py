"""Misc. testing for kernel troubleshooting."""

from time import perf_counter

import numpy as np
import pandas as pd

import seaborn as sns
sns.set_theme()

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

# For comparison
from sklearn.svm import SVC, LinearSVC

# Local import
from svm import MultiGDSVM
import dim_reduction

def main():
    path = "data/TCGA-PANCAN-HiSeq-801x20531/"
    raw = pd.read_csv(path + "data.csv").iloc[:, 1:]
    labels = pd.read_csv(path + "labels.csv").iloc[:, 1:]

    # Merge X and y
    raw = pd.concat([raw, labels], axis=1)

    # Move class to front
    col = raw.pop("Class")
    raw.insert(0, col.name, col)

    X = raw.iloc[:, 1:]
    y = raw.iloc[:, 0]

    # Scale input features
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    # Encode target labels
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

    #X = dim_reduction.CUR(1000).fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

    svm = MultiGDSVM(kernel="linear", C=1, gamma=1, n_dims=1_000, dim_method="grp")

    t0 = perf_counter()

    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)

    print(f"time: {perf_counter() - t0:.4f}")
    print(f"Accuracy: {(y_pred == y_test).sum() / y_pred.shape[0]:.4f}")

if __name__ == "__main__":
    main()
