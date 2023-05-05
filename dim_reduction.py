"""Dimensionality reduction transformers."""

import warnings

import numpy as np

class GaussianRandProj:
    def __init__(self, num_features):
        self.num_features = num_features

    def fit(self, X):
        self.proj_matrix_ = np.random.normal(size=(X.shape[1], self.num_features))

    def transform(self, X):
        return np.dot(X, self.proj_matrix_)

    def fit_transform(self, X):
        self.fit(X)

        return self.transform(X)


class RFF:
    def __init__(self, num_features, gamma):
        self.num_features = num_features
        self.gamma = gamma

    def fit(self, X):
        n_features = X.shape[1]

        self.omega_ = np.random.normal(size=(n_features, self.num_features))
        self.b_ = 2 * np.pi * np.random.rand(1, self.num_features)

    def transform(self, X):
        X = np.sqrt(2 * self.gamma / self.num_features) * np.cos(X @ self.omega_ + self.b_)

        return X

    def fit_transform(self, X):
        self.fit(X)

        return self.transform(X)

class PCA:
    def __init__(self, num_features):
        self.num_features = num_features

    def fit(self, X):
        # Subtract the mean from the data
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        # Compute covariance matrix
        cov_mat = np.cov(X_centered, rowvar=False)

        # Compute eigenvectors and eigenvalues
        eigenvals, self.eigenvecs_ = np.linalg.eigh(cov_mat)

        # Sort eigenvectors in descending order by eigenvalue
        idx = eigenvals.argsort()[::-1]
        self.eigenvecs_ = self.eigenvecs_[:,idx]
        eigenvals = eigenvals[idx]

        # Get top eigenvectors
        self.eigenvecs_ = self.eigenvecs_[:, :self.num_features]

    def transform(self, X):
        # Transform data using eigenvectors
        X = (X - self.mean_) @ self.eigenvecs_

        return X

    def fit_transform(self, X):
        self.fit(X)

        return self.transform(X)
    
class CUR:
    """Must be applied to whole dataset ahead of time."""
    def __init__(self, num_features):
        self.result_features = num_features

    def fit_transform(self, X):
        result_features = self.result_features

        if result_features > X.shape[0]:
            result_features = X.shape[0]
            warnings.warn(f"Target dimensionality < sample count, lowering to match: {X.shape[0]}")

        # Sample columns and rows of the original matrix
        col_norms = np.sum(X ** 2, axis=0)
        col_prob = col_norms / np.sum(col_norms)
        cols = np.random.choice(X.shape[1], size=result_features, replace=False, p=col_prob)
        C = X[:, cols] / np.sqrt(result_features * col_prob[cols])
        
        # Compute the C and R matrices
        row_norms = np.sum(X ** 2, axis=1)
        row_prob = row_norms / np.sum(row_norms)
        rows = np.random.choice(X.shape[0], size=result_features, replace=False, p=row_prob)
        R = X[rows, :] / np.sqrt(result_features * row_prob[rows, np.newaxis])
        
        # Compute the U matrix
        U, _, _ = np.linalg.svd(C, full_matrices=False)
        U = U[:, :result_features] / np.sqrt(result_features * col_prob[cols])
        
        # Compute the low-rank approximation
        V, _, _ = np.linalg.svd(R, full_matrices=False)
        sigma = np.diag(np.sqrt(col_norms[cols])[:result_features])
        CUR = U @ sigma @ V
        
        return CUR
