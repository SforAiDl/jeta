from typing import List, Tuple

import flax.linen as nn


def MLP(dims: List[int], activations="relu") -> nn.Module:
    """Create a multi-layer perceptron (MLP) module.
    This consists of only Dense Layers followed by a relu activation.

    Args:
        dims (List[int]): dimentions of the MLP Layers.
        activations (str, optional): Activation function to be used (relu, leaky relu, softmax, tanh). Defaults to 'relu'.

    Returns:
        nn.Module: Returns a Flax Module that can be used as a MLP.
    """

    if activations.lower() == "relu":
        activation = nn.relu
    elif activations.lower() == "leaky relu":
        activation = nn.leaky_relu
    elif activations.lower() == "softmax":
        activation = nn.softmax
    elif activations.lower() == "tanh":
        activation = nn.tanh
    else:
        raise ValueError(
            f"Invalid activation function\nExpected: (relu, leaky relu, softmax, tanh) but got: {activations}"
        )

    class Model(nn.Module):
        @nn.compact
        def __call__(self, x, train=True):
            for dim in dims[:-1]:
                x = nn.Dense(dim)(x)
                x = activation(x)
            x = nn.Dense(dims[-1])(x)
            return x

    mlp = Model()
    return mlp


def ConvBlock(
    channels: int, kernel_size: Tuple[int], strides: Tuple[int], padding: str
) -> nn.Module:
    """Create a convolutional block module.
    This consists of a convolution layer, batch normalization and a relu activation.

    Args:
        channels (int): Number of filters in the convolution layer.
        kernel_size (Tuple[int]): Size of the convolution kernel.
        strides (Tuple[int]): Strides of the convolution kernel.
        padding (str): Padding of the convolution.

    Returns:
        nn.Module: Returns a Flax Module that can be used as a convolutional block.
    """

    class Model(nn.Module):
        @nn.compact
        def __call__(self, x, train=True):
            x = nn.Conv(channels, kernel_size, strides, padding)(x)
            x = nn.BatchNorm(use_running_average=not train)(x)
            x = nn.relu(x)
            return x

    conv_block = Model()
    return conv_block


def CNN(dims: List[int]) -> nn.Module:
    """Create a convolutional neural network (CNN) module.
    Each Convolution block consists of a convolution layer, batch normalization and a relu activation.

    Args:
        dims (List[int]): Number of filters in each CNN layer.

    Returns:
        nn.Module: Returns a Flax Module that can be used as a CNN.
    """

    class Model(nn.Module):
        @nn.compact
        def __call__(self, x, train=True):
            for dim in dims:
                x = ConvBlock(dim, (3, 3), (1, 1), "SAME")(x, train)
            return x

    cnn = Model()
    return cnn


def RNN(dims: List[int], type: str = "lstm", activation: str = "relu") -> nn.Module:
    """Create a recurrent neural network (RNN) module.
    This consists of a Recurrent(LSTM, GRU) layer followed by a relu activation.

    Args:
        dims (List[int]): Number of units in the Recurrent layer.
        type (str, optional): Type of the Recurrent layer(lstm, rnn). Defaults to 'lstm'.
        activation (str, optional): Activation function to be used (relu, leaky relu, softmax, tanh). Defaults to 'relu'.

    Returns:
        nn.Module: Returns a Flax Module that can be used as a RNN.
    """

    if type.lower() == "lstm":
        Cell = nn.OptimizedLSTMCell
    elif type.lower() == "gru":
        Cell = nn.GRUCell
    else:
        raise ValueError(f"Invalid type\nExpected: (lstm, gru) but got: {type}")

    if activation.lower() == "relu":
        activation = nn.relu
    elif activation.lower() == "leaky relu":
        activation = nn.leaky_relu
    elif activation.lower() == "softmax":
        activation = nn.softmax
    elif activation.lower() == "tanh":
        activation = nn.tanh
    else:
        raise ValueError(
            f"Invalid activation function\nExpected: (relu, leaky relu, softmax, tanh) but got: {activation}"
        )

    class Model(nn.Module):
        @nn.compact
        def __call__(self, x, train=True):
            for dim in dims[:-1]:
                x = Cell(dim)(x)
                x = activation(x)
            x = nn.Cell(dims[-1])(x)
            return x

    rnn = Model()
    return rnn
