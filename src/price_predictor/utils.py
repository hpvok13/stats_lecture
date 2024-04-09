import torch.nn as nn


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self
        for k, v in self.items():
            if isinstance(v, dict):
                self[k] = AttrDict(v)


def get_activation_class(activation):
    if activation == "relu":
        return nn.ReLU
    elif activation == "tanh":
        return nn.Tanh
    elif activation == "sigmoid":
        return nn.Sigmoid
    elif activation == "softplus":
        return nn.Softplus
    elif activation == "softsign":
        return nn.Softsign
    elif activation == "elu":
        return nn.ELU
    elif activation == "selu":
        return nn.SELU
    elif activation == "gelu":
        return nn.GELU
    elif activation == "leaky_relu":
        return nn.LeakyReLU
    else:
        raise ValueError(f"Activation function {activation} not supported.")
