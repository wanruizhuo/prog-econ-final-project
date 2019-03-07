import pandas as pd
import numpy as np

from src.model_code.panel_model import gen_xy, Q_transform_matrix
from src.model_code.panel_model import fixed_effects_model, random_effects_model
from bld.project_paths import project_paths_join as ppj

data = pd.read_csv(ppj("OUT_DATA", "productivity_clean.csv"))

def hausman():
        
    X, y = gen_xy()
    Q, P = Q_transform_matrix()
    sigma_v_sqaured_hat_hat = fixed_effects_model(Q, X, y)[2]
    
    beta_GLS_hat = random_effects_model(P, Q, X, y, sigma_v_sqaured_hat_hat)[0]
    
    beta_Within_tilde = fixed_effects_model(Q, X, y)[0]
    
    var_beta_GLS = random_effects_model(P, Q, X, y, sigma_v_sqaured_hat_hat)[1]
    
    var_beta_tilde = fixed_effects_model(Q, X, y)[1]
    
    # Calculate q_1 for further test statistic computation
    q_1_hat = beta_GLS_hat - beta_Within_tilde
    
    # Calculate variance of q_1
    var_q_1_hat = var_beta_tilde - var_beta_GLS
    
    # Calculate the test statistic for Hausman test
    m_1 = q_1_hat.T @ np.linalg.inv(var_q_1_hat) @ q_1_hat
    
    # Set up critical value 11.143 for Hausman test at 2.5% level
    if(m_1 > 11.143):
        print("Null hypothesis can be rejected.")
    else:
        print("Null hypothesis cannot be rejected.")

hausman()