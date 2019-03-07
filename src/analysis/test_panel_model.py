"""This function is used for testing panel_data_regression_model function.                          
"""

import sys
import pytest
import pandas as pd
import numpy as np

#from numpy.testing import assert_array_equal
from src.model_code.panel_model import gen_xy, Q_transform_matrix, fixed_effects_model, random_effects_model
from bld.project_paths import project_paths_join as ppj


# load and prepare data
data = pd.read_csv(ppj("OUT_DATA", "productivity_clean.csv"))

X, y = gen_xy()
Q, P = Q_transform_matrix()
sigma_v = fixed_effects_model(Q, X, y)[2]


@pytest.fixture
def setup_fixed_model():
    """
    Assigns elements of calculation process.
    """
    out = {}
    out['Q'] = Q
    out['X'] = X
    out['y'] = y

    return out

@pytest.fixture
def setup_random_model():
    """ """
    out = {}
    out['P'] = P
    out['Q'] = Q
    out['X'] = X
    out['y'] = y
    out['sigma_v_squared_hat_hat'] = sigma_v
    
    return out

@pytest.fixture
def expected_panel_model():
    """
    Calculates expected results of coefficient.
    """
    out = {}
    out['beta_Within_tilde'] = np.array([-0.026, 0.292, 0.768, -0.005])
    out['var_beta_tilde'] = np.array([
                                 [8.410907e-04, -1.318886e-04, -3.765249e-04, -1.141845e-05],
                                 [-1.318886e-04, 6.309980e-04, -5.662002e-04, -1.099228e-05],
                                 [-3.765249e-04, -5.662002e-04, 9.055127e-04, 1.653519e-05],
                                 [-1.141845e-05, -1.099228e-05, 1.653519e-05, 9.775783e-07],
                             ])
    out['beta_GLS_hat'] = np.array([0.004, 0.311, 0.730, -0.006])
    out['var_beta_GLS_hat'] = np.array([
                                   [5.445312e-04, -5.599064e-05, -3.428480e-04, -6.915298e-06],
                                   [-5.599064e-05, 3.894819e-04, -3.276008e-04, -7.759666e-06],
                                   [-3.428480e-04, -3.276008e-04, 6.166693e-04, 1.129612e-05],
                                   [-6.915298e-06, -7.759666e-06, 1.129612e-05, 8.173974e-07],
                               ])

    return out

#def test_fast_batch_update_mean(setup_update, expected_update):
    """
    Assertion on calculated and expected means.
    """
    #calc_mean, calc_root_cov = fast_batch_update(**setup_update)
    #assert np.allclose(calc_mean, expected_update['mean'])


def test_panel_model_beta_Within(setup_fixed_model, expected_panel_model):
    """
    Assertion on calculated and expected means.
    """
    calc_beta_Within_tilde, calc_var_beta_tilde, calc_sigma_v = fixed_effects_model(**setup_fixed_model)
    assert np.allclose(calc_beta_Within_tilde.round(3), expected_panel_model['beta_Within_tilde'])


def test_panel_model_var_Within_tilde(setup_fixed_model, expected_panel_model):
    """
    Assertion on calculated and expected covariances.
    """
    calc_beta_Within_tilde, calc_var_beta_tilde, calc_sigma_v = fixed_effects_model(**setup_fixed_model)
    assert np.allclose(calc_var_beta_tilde, expected_panel_model['var_beta_tilde'])


def test_panel_model_beta_GLS(setup_random_model, expected_panel_model):
    calc_beta_GLS_hat, calc_var_beta_GLS = random_effects_model(**setup_random_model)
    assert np.allclose(calc_beta_GLS_hat.round(3), expected_panel_model['beta_GLS_hat'])


def test_panel_model_var_beta_GLS(setup_random_model, expected_panel_model):
    calc_beta_GLS_hat, calc_var_beta_GLS = random_effects_model(**setup_random_model)
    assert np.allclose(calc_var_beta_GLS, expected_panel_model['var_beta_GLS_hat'])


if __name__ == "__main__":
    status = pytest.main([sys.argv[0]])
    sys.exit(status)
