import torch
import torch.nn as nn
import torch.optim as optim

from utils.dicts import *


def get_torch_device(cuda=True):

    if torch.cuda.is_available() and cuda:
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def get_pytorch_activation(activation):

    if activation == ActivationDict.relu:
        activation = nn.ReLU()
    elif activation == ActivationDict.sigmoid:
        activation = nn.Sigmoid()
    elif activation == ActivationDict.tanh:
        activation = nn.Tanh()
    else:
        print("Activation '{}' is not defined yet. Set to default Sigmoid activation.".format(activation))
        activation = nn.Sigmoid()

    return activation


def get_pytorch_optimizer(optimizer, parameters, learning_rate):

    if optimizer == OptimizerDict.sgd:
        optimizer = optim.SGD(parameters, lr=learning_rate)
    elif optimizer == OptimizerDict.adam:
        optimizer = optim.Adam(parameters, lr=learning_rate)
    elif optimizer == OptimizerDict.rmsprop:
        optimizer = optim.RMSprop(parameters, lr=learning_rate)
    else:
        print("Optimizer '{}' is not defined yet. Set to default SGD optimizer.".format(optimizer))
        optimizer = optim.SGD(parameters, lr=learning_rate)

    return optimizer


def get_pytorch_criterion(criterion):

    if criterion == LossDict.mse:
        criterion = nn.MSELoss()
    elif criterion == LossDict.cross_entropy:
        criterion = nn.CrossEntropyLoss()
    elif criterion == LossDict.binary_cross_entropy:
        criterion = nn.BCELoss()
    else:
        print("Loss '{}' is not defined yet. Set to default CrossEntropyLoss optimizer.".format(criterion))
        criterion = nn.CrossEntropyLoss()

    return criterion

