from abc import ABC

from jax import random


class BaseLearner(ABC):
    """Base Learner class

    Attributes:
        batch_size(int): batch size for training the algorithm on
        alpha(float): Learning rate
        fas(int): Fast adaptation steps
        seed(int): integer value for generating key for random generators
        first_order(bool): True/False, for extending MAML to FO-MAML

    """

    def __init__(
        self, batch_size: int = 32, alpha: float = 0.001, fas: int = 1, **kwargs
    ):

        self.batch_size = batch_size
        self.alpha = alpha
        self.fas = fas
        self.losses = []
        self.seed = kwargs["seed"] if "seed" in kwargs else None
        self.first_order = kwargs["first_order"] if "first_order" in kwargs else None
        if self.seed is not None:
            self.key = random.PRNGKey(self.seed)

    def train(self, params, batch):
        """train algorithm

        Args:
            params(ndarray): initial parameters for the model
            batch(ndarray): training batch for the algorithm

        Return:
            new_params(ndarray): updated parameters after 1 epoch
        """
        raise NotImplementedError

    def loss(self, params, batch):
        """loss with current parameters

        Args:
            params(ndarray): current set of parameters
            batch(ndarray): training batch for the algorithm

        Return:
            loss(float): loss incurred
        """
        raise NotImplementedError

    def test(self, params, batch):
        """test algorithm

        Args:
            params(ndarray): final set of parameters after training
            batch(ndarray): testing batch of data

        Return:
            output(ndarray): predicted output
        """
        raise NotImplementedError
