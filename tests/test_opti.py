import unittest

import sys

sys.path.append('C:users/riddh/git-repos/jeta/jeta')
# from opti_trainer import *
import opti_trainer as opt
import flax

class Test_OptiTrainer(unittest.TestCase):
    Opti = opt.OptiTrainer

    def test_create(self, params, apply_fn, adapt_fn, loss_fn, tx):
        # self.assertIsInstance(params,)
        self.assertIsInstance(apply_fn, Callable)
        self.assertIsInstance(adapt_fn, Callable)
        self.assertIsInstance(loss_fn, Callable)
        state = Opti.create(params, apply_fn, adapt_fn, loss_fn, tx)
        self.assertIsInstance(state, MetaTrainState)

    def test_meta_train_step(self, state: MetaTrainState,
                             tasks,
                             metrics: List[Callable[[ndarray, ndarray], ndarray]] = []):
        self.assertIsInstance(state, MetaTrainState)
        self.assertIsInstance(metrics, List[Callable[[ndarray, ndarray], ndarray]])
        meta_train = Opti.meta_train_step(state, tasks, metrics)
        self.assertIsInstance(meta_train, Tuple[MetaTrainState, ndarray, List[ndarray]])

    def test_meta_test_step(self, state, tasks, metrics):
        self.assertIsInstance(state, MetaTrainState)
        self.assertIsInstance(metrics, List[Callable[[ndarray, ndarray], ndarray]])
        meta_test = Opti.meta_test_step(state, tasks, metrics)
        self.assertIsInstance(meta_test, Tuple[jnp.ndarray, List[jnp.ndarray]])

    def test_meta_loss(self, params, apply_fn, adapt_fn, loss_fn, task, metrics):
        meta_loss = Opti.meta_loss(params, apply_fn, adapt_fn, loss_fn, task, metrics)
        self.assertIsInstance(meta_loss, Tuple[ndarray, List[ndarray]])

if __name__ == '__main__':
    unittest.main()
