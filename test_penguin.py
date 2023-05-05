"""Testing multiclass SVM with archipelago penguins data."""

import time

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

# Local import
from svm import BinaryGDSVM, MultiGDSVM
import dim_reduction

from sklearn.svm import SVC

def main():
    # Get normalized and encoded data
    penguins, X, y = get_data()
    y = y + 1

    # Split data (80:20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    for name in ["linear", "poly", "rbf"]:
        print(name)

        # Create model
        svm = create_model("rbf")
        #svm = SVC(kernel="poly")

        t0 = time.perf_counter()

        # Fit model and make predictions
        svm.fit(X_train, y_train)
        y_pred = svm.predict(X_test)

        # Evaluate and print results
        print(f"time: {time.perf_counter() - t0:.4f}")
        print(f"sup vecs: {np.sum([clf.support_vectors_.shape[0] for clf in svm.classifiers_.values()])}")
        print(f"accuracy: {np.sum(y_pred == y_test) / y_pred.shape[0]:.4f}")

        print()

    # Plot data if wanted
    pairplot(penguins)


def get_data():
    # Load data
    penguins = sns.load_dataset("penguins").dropna()
    X = penguins.loc[:, ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]].values
    y = penguins.loc[:, "species"]

    # Scale input features
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    # Encode target labels
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

    return penguins, X, y


def create_model(kernel):
    if kernel == "linear":
        return MultiGDSVM(kernel="linear", C=1)
    elif kernel == "poly":
        return MultiGDSVM(kernel="poly", C=1, degree=5, gamma=1)
    elif kernel == "rbf":
        return MultiGDSVM(kernel="rbf", C=1, gamma=1)#, dim_method="rff", n_dims=3)
    
    raise ValueError(f"Invalid kernel name '{kernel}'.")


def pairplot(penguins):
    sns.set_theme()

    penguins = penguins.loc[:, ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g", "species"]].dropna()

    sns.pairplot(penguins, hue="species")#, corner=True)

    title = "Archipelago Penguins"

    plt.suptitle(title)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
