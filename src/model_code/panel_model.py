import numpy as np
import pandas as pd

from bld.project_paths import project_paths_join as ppj

N = 48
T = 17
K = 4

data = pd.read_csv(ppj("OUT_DATA", "productivity_clean.csv"))

def gen_xy():
    """Getting dependent and independent variables.
    
       Returns:
           x (np.ndarray): Independent variables
           y (np.ndarray): Dependent variable
    """
    
    y = data.iloc[:, -1].values
    
    x_temp1 = data.iloc[:, -4:].drop(columns = ['LNGSP'])
    x_um = data['UNEMP']
    x = x_temp1.join(x_um)
    x = x.values
    
    return x, y
    

def Q_transform_matrix():
    """Obtaining transform matrix and projection matrix.
    
       Returns:
           Q (np.arrays): Transform matrix which equals to (I_NT - P)
           P (np.arrays): Projection matrix which will be used for
           demeaning regressions
    """
    I_N = np.identity(N)
    J_T = np.ones([T, 1])
    Z_mu = np.kron(I_N, J_T)
    P = Z_mu @ np.linalg.inv(Z_mu.T @ Z_mu) @ Z_mu.T
    I_NT = np.identity(N * T)
    Q = I_NT-P
    
    return Q, P


def fixed_effects_model(Q, X, y):
    """Constructing fixed effects model regression.
    
       Args:
           Q (np.ndarray): Transform matrix which equals to I_NT - P
           y (np.ndarray): Dependent variable
           X (np.ndarray): Independent variable
           
       Returns:
           beta_Within_tilde (np.ndarray): Within estimator for fixed effects
           model
           var_beta_tilde (np.ndarray): Variance of within estimator for fixed
           effects model
           sigma_v_squared_hat_hat (float64): Estimation for variance-covariance
           components of error term

    """
    beta_Within_tilde = np.linalg.inv(X.T @ Q @ X) @ X.T @ Q @y
    sigma_v_squared_hat_hat = (y.T @ Q @ y - y.T @ Q @ X @ np.linalg.inv(X.T @ Q @ X) @ X.T @ Q @ y)/(N * (T - 1) - K)
    var_beta_tilde = sigma_v_squared_hat_hat * np.linalg.inv(X.T @ Q @ X)
    
    return beta_Within_tilde, var_beta_tilde, sigma_v_squared_hat_hat


def random_effects_model(P, Q, X, y, sigma_v_squared_hat_hat):
    """Constructing random effects model regression.
    
       Args:
           Q (np.ndarray): Transform matrix which equals to (I_NT - P)
           P (np.ndarray): Projection matrix which will be used for demeaning
           regressions
           X (np.ndarray): Independent variable
           y (np.ndarray): Dependent variable
           sigma_1_squared_hat_hat (float64): Estimation for variance-covariance
           components of error term
           
       Returns:
           beta_GLS_hat: GLS estimator for random effects model
           var_beta_GLS_hat: Variance of GLS estimator for random effects model
    """
    iota_NT = np.ones([N * T, 1])
    Z = np.hstack((iota_NT, X))
    sigma_1_squared_hat_hat = (y.T @ P @ y - y.T @ P @ Z @ np.linalg.inv(Z.T @ P @ Z) @ Z.T @ P @ y)/(N - K - 1)
    phi_squared_hat = sigma_v_squared_hat_hat/sigma_1_squared_hat_hat
    J_NT_bar = (iota_NT @ iota_NT.T)/(N * T)
    B_XX = X.T @ (P - J_NT_bar) @ X
    B_Xy = X.T @ (P - J_NT_bar) @ y
    W_XX = X.T @ Q @ X
    W_Xy = X.T @ Q @ y
    beta_GLS_hat = np.linalg.inv(W_XX + phi_squared_hat * B_XX) @ (W_Xy + phi_squared_hat * B_Xy)
    var_beta_GLS = sigma_v_squared_hat_hat * np.linalg.inv(W_XX + phi_squared_hat * B_XX)
    
    return beta_GLS_hat, var_beta_GLS
