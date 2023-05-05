"""Experimental SVM classifier using stochastic or mini-batch subgradient method.
    
Uses subgradient method for the dual formulation of the SVM problem to...
    * Support non-linear kernels (polynomial and Gaussian Radial Basis Function (RBF))
    * Potentially support scalability compared to quadratic prog approach

TODO: Update MultiGDSVM to compute kernel once rather than for each classifier
    * Toggleable for when Nystrom approx used
    * No downsides when nystrom approx not used
"""

from abc import ABC
import warnings

import numpy as np

# Local
import dim_reduction


class BaseGDSVM(ABC):
    """Abstract base to be inherited from for SVM implementations. Compatible w/ sklearn CV."""
    def __init__(self, C=1.0, l1_reg=0.0, 
                 kernel="linear", degree=3, gamma=1, 
                 nystrom_rows=200, n_dims=0, dim_method="pca",
                 iters=10_000, eps=1e-10, batch_size=32, shrink_thresh=1e-6,
                 lr=0.01, beta1=0.9, beta2=0.999, 
                 verbose=True):
        # Regularization
        self.C = C             # Regularization parameter
        self.l1_reg = l1_reg   # Factor for additional L1 regularization to encourage fewer SVs

        # Kernel
        self.kernel = kernel  # Kernel type: {"linear", "poly", "rbf"}
        self.degree = degree  # Polynomial degree (if kernel is 'poly')
        self.gamma = gamma    # Kernel coefficient (if kernel is 'rbf', 'poly')

        # Built-in dimensionality reduction
        self.nystrom_rows = nystrom_rows  # Num samples for Nystrom approximation
        self.n_dims = n_dims              # Num features after dimensionality reduction (<1 ~= none)
        self.dim_method = dim_method      # Dimensionality reduction ('rff', 'pca', 'grp')

        # Optimization
        self.iters = iters                  # Max iterations
        self.eps = eps                      # Stopping tolerance
        self.batch_size = batch_size        # Num samples for gradient (defaults to 1 := SGD)
        self.shrink_thresh = shrink_thresh  # Alpha threshold for rejecting SV
        self.lr = lr                        # Learning rate
        self.beta1 = beta1                  # First ADAM param
        self.beta2 = beta2                  # Second ADAM param

        self.verbose = verbose

    def get_params(self, deep=True):
        return {
            "C": self.C, "l1_reg": self.l1_reg,
            "kernel": self.kernel, "degree": self.degree, "gamma": self.gamma,
            "batch_size": self.batch_size, 
            "nystrom_rows": self.nystrom_rows, 
            "n_dims": self.n_dims, "dim_method": self.dim_method,
            "iters": self.iters, "eps": self.eps,
            "lr": self.lr, "beta1": self.beta1, "beta2": self.beta2,
            "shrink_thresh": self.shrink_thresh,
            "verbose": self.verbose,
        }
    
    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)

        return self


class BinaryGDSVM(BaseGDSVM):
    """Binary SVM classifier."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def fit(self, X, y, pos_label=1):
        self.kernel = self.kernel.lower()

        # Perform dimensionality reduction
        if self.n_dims > 0:
            self.dim_method = self.dim_method.lower()
            self.dim_reducer_ = self._create_dim_reducer()
            X = self.dim_reducer_.fit_transform(X)

        # Compute and save approximated kernel matrix for nystrom rows
        K = self._kernel_matrix_nystrom(X)

        # Convert labels to -1 or 1
        y = np.where(y == pos_label, 1, -1)

        # Prepare for optimization with ADAM
        num_samples = X.shape[0]
        alpha = np.full(num_samples, self.shrink_thresh * 10)  # Lagrange multipliers
        m = np.zeros(num_samples)
        v = np.zeros(num_samples)
        t = 0

        # Perform optimization with subgradient method for dual formulation
        for epoch in range(self.iters):
            prev_alpha = alpha.copy()

            # Get indices of random sample for stochastic/mini-batch
            batch_indices = np.random.permutation(num_samples)[:self.batch_size]

            for i in batch_indices:
                # Shrinking (leave out zero'd support vector candidates)
                if alpha[i] >= self.shrink_thresh:
                    # Get gradient of L wrt alpha
                    grad = 1 - y[i] * np.sum(alpha * y * K[:, i])

                    # Punish non-zero lagrangian multipliers to encourage sparsity
                    grad -= self.l1_reg * np.abs(alpha[i])

                    # Update moment estimations for ADAM
                    t += 1
                    m[i] = self.beta1 * m[i] + (1 - self.beta1) * grad
                    v[i] = self.beta2 * v[i] + (1 - self.beta2) * grad ** 2
                    m_hat = m[i] / (1 - self.beta1 ** t)
                    v_hat = v[i] / (1 - self.beta2 ** t)

                    # Perform update
                    alpha[i] += self.lr * m_hat / (np.sqrt(v_hat) + 1e-8)

                    # Restrict alpha to [0, C]
                    alpha[i] = max(0, min(self.C, alpha[i]))

            # Break if converged based on tolerance
            if np.linalg.norm(alpha - prev_alpha) < self.eps:
                if self.verbose:
                    print(f"converged (epoch={epoch})")
                break

        # Determine support vectors w/ non-zero lagrange multipliers
        sv_indices = np.where(alpha > 1e-6)[0]
        self.support_vectors_ = X[sv_indices]
        self.alpha_ = alpha[sv_indices]
        self.y_ = y[sv_indices]

        if self.alpha_.size == 0:
            warnings.warn("No support vectors learned")

        # Compute additional values needed for prediction
        if self.kernel == 'linear':
            # w = sum(alpha * y * kern(x)), b = mean(y - w^T * x)
            self.w_ = np.sum(self.alpha_.reshape(-1, 1) * self.y_.reshape(-1, 1) * self.support_vectors_, axis=0)
            self.b_ = np.mean(self.y_ - np.dot(self.support_vectors_, self.w_))
        else:
            K = self._kernel_matrix_nystrom(self.support_vectors_)
            self.b_ = np.mean(self.y_ - np.dot(K, self.alpha_ * self.y_))

    def predict(self, X, return_score=False):
        if self.n_dims > 0:
            X = self.dim_reducer_.transform(X)

        if return_score:
            return self._decision_function(X)

        return np.sign(self._decision_function(X)).astype(int)

    def _kernel_matrix(self, X1, X2):
        if self.kernel == 'linear':
            return X1 @ X2.T
        elif self.kernel == 'poly':
            # TODO: parameterize intercept (0 for now)
            return (self.gamma * (X1 @ X2.T) + 0.0) ** self.degree
        elif self.kernel == 'rbf':
            dists = np.sum((X1[:, np.newaxis] - X2) ** 2, axis=2)
            return np.exp(-self.gamma * dists)
        
        raise ValueError(f"Invalid kernel name '{self.kernel}'")
    
    def _kernel_matrix_nystrom(self, X):
        if self.nystrom_rows <= 0 or self.nystrom_rows >= X.shape[0]:
            return self._kernel_matrix(X, X)

        # Get sample matrix w/ uniform sampling
        n_samples = X.shape[0]
        sample_size = self.nystrom_rows
        indices = np.random.permutation(n_samples)[:sample_size]
        X_sub = X[indices]

        # Use X_sub to compute approximated kernel
        K_mm = self._kernel_matrix(X_sub, X_sub)# + np.eye(X_sub.shape[0]) * 1e-6
        K_nm = self._kernel_matrix(X, X_sub)
        K = K_nm @ np.linalg.pinv(K_mm) @ K_nm.T
        
        return K
    
    def _create_dim_reducer(self):
        if self.dim_method == "grp":
            return dim_reduction.GaussianRandProj(self.n_dims)
        elif self.dim_method == "rff":
            return dim_reduction.RFF(self.n_dims, self.gamma)
        elif self.dim_method == "pca":
            return dim_reduction.PCA(self.n_dims)

        raise ValueError(f"Invalid dimension reduction name '{self.dim_method}'")
    
    def _decision_function(self, X):
        if self.kernel == 'linear':
            return np.dot(X, self.w_) + self.b_
        else:
            K = self._kernel_matrix(X, self.support_vectors_)
            return np.dot(K, self.alpha_ * self.y_) + self.b_
    

class MultiGDSVM(BaseGDSVM):
    """Multiclass SVM classifier using a One-vs-All (OvA) approach."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.classifiers_ = dict()

        for target in self.classes_:
            # Create binary classifier
            self.classifiers_[target] = BinaryGDSVM(**self.get_params())

            # Fit using correct label
            self.classifiers_[target].fit(X, y, pos_label=target)

    def predict(self, X):
        scores = []

        # Get scores per sample for each binary classifier
        for target in self.classes_:
            clf = self.classifiers_[target]
            score = clf.predict(X, return_score=True)
            scores.append(score)

        # Get indices largest absolute score
        label_idx = np.argmax(scores, axis=0)

        # Get class label based on index
        labels = np.take(self.classes_, label_idx)

        return labels
