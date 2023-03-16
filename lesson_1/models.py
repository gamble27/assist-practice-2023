import numpy as np
from statsmodels.multivariate import pca
from sklearn.linear_model import LinearRegression


def estim_ls(x,y):
    x1 = np.array(x)
    y1 = np.array(y)
    try:
        return np.linalg.inv(x1.T @ x1) @ x1.T @ y1
    except np.linalg.LinAlgError:
        return (1 / (x1.T @ x1)) * (x1.T @ y1)


def estim_ridge(x,y,l):
    x1 = np.array(x)
    y1 = np.array(y)
    try:
        return np.linalg.inv(
            x1.T @ x1 + l * np.ones(x1.shape[1])
        ) @ x1.T @ y1
    except np.linalg.LinAlgError:
        return (1 / (x1.T @ x1 + l)) * (x1.T @ y1)


def estim_gd_step(
        x: np.ndarray,
        y: np.ndarray,
        alpha: float,
        beta: np.ndarray
):
    return beta - (alpha/x.shape[0]) * x.T @ (x @ beta - y)



def estim_gd(x,y,alpha,n):
    # randomly init beta
    beta = np.random.normal(0, 1, x.shape[1])

    # run training cycle
    for _ in range(n):
        beta = estim_gd_step(x,y,alpha, beta)

    return beta


class PCA:
    def __init__(self, x):
        self.x = x
        self.pc = pca.PCA(x, method='eig')
        self.pc_fit = None
        self.lr_pca = None

    def get_eigenvalues(self):
        return self.pc.eigenvals

    def plot_eigenvalues(self):
        self.pc.plot_scree(log_scale=False)

    def transform_data(self, n_comp):
        self.pc_fit = pca.PCA(self.x, method='eig', ncomp=n_comp)

    def get_transformed_data(self):
        assert self.pc_fit
        return self.pc_fit.transformed_data

    def fit_regression(self, y):
        self.lr_pca = LinearRegression(fit_intercept=True).fit(self.x, y)

    def get_y_hat(self):
        assert self.lr_pca
        return self.lr_pca.predict(self.x)
