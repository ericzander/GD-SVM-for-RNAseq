"""Testing binary SVM with Old Faithful geyser data."""

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV

# Local import
from svm import BinaryGDSVM#, TransductiveGDSVM


def main():
    # Get normalized and encoded data
    X, y = get_data()

    # Split data (80:20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)

    # Create model
    svm = create_model("linear")

    # Fit and predict
    svm.fit(X_train, y_train)
    #svm.fit(X_train, y_train, X_test)
    y_pred = svm.predict(X_test)
    y_pred = np.where(y_pred == 1, 1, 0)

    # Evaluate
    print(f"accuracy: {np.sum(y_pred == y_test) / y_pred.shape[0]:.4f}")

    # Plot data if wanted
    plot_data(X, y, svm)


def get_data():
    # Load data
    geyser = sns.load_dataset("geyser")
    X = geyser.iloc[:, :-1].values
    y = geyser.iloc[:, -1]

    # Scale input features
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    # Encode target labels
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

    return X, y


def create_model(kernel):
    if kernel == "linear":
        return BinaryGDSVM(kernel="linear", C=0.02)
    elif kernel == "poly":
        return BinaryGDSVM(kernel="poly", C=1, degree=3, gamma=1.5)
    elif kernel == "rbf":
        return BinaryGDSVM(kernel="rbf", C=0.1, gamma=100)
    
    raise ValueError(f"Invalid kernel name '{kernel}'.")


def plot_data(X, y, svm=None):
    sns.set_theme()

    plt.figure(figsize=(6, 6), dpi=150)

    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, palette="Set1")

    title = "Old Faithful Eruption Duration"

    if svm is not None:
        # Plot support vectors
        sv = svm.support_vectors_
        plt.scatter(sv[:, 0], sv[:, 1], color="white", s=4, label="Support Vectors")

        x_min, x_max = -0.2, 1.2
        y_min, y_max = -0.2, 1.2
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.005),
                            np.arange(y_min, y_max, 0.005))
        grid = np.c_[xx.ravel(), yy.ravel()]

        # Predict class labels for the grid
        Z = svm.predict(grid)
        Z = Z.reshape(xx.shape)

        # Plot the grid points with predicted class labels
        plt.contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha=0.3)

        # Plot the decision boundary
        #plt.contour(xx, yy, Z, colors='k',)# levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])

        title += f"\n<kernel={svm.kernel}, C={svm.C}"
        if svm.kernel in ["poly", "rbf"]:
            if svm.kernel == "poly":
                title += f", degree={svm.degree}"
            title += f", gamma={svm.gamma}"
        title += ">"
    
    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.1, 1.1)
    plt.title(title)
    plt.xlabel("duration")
    plt.ylabel("waiting")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
