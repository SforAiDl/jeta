import sys

sys.path.append('C:users/riddh/git-repos/jeta/jeta')
import maml

def test_maml_adapt(params, apply_fn, loss_fn, support_set):
    theta = params["params"]
    theta_new = maml.maml_adapt(params, apply_fn, loss_fn, support_set)
    assert theta != theta_new