import torch
import torch.nn as nn
from torch.autograd import Variable

from utils.nn_func import *
from utils.nn_dict import *


class LinReg(nn.Module):
    """
    Linear Regression Class
    """

    def __init__(self, input_dim,
                 criterion=LossDict.mse, optimizer=OptimizerDict.sgd,
                 learning_rate=0.01):

        # calling the super class
        super(LinReg, self).__init__()

        # initializing the linear layer
        self.linear_layer = nn.Linear(input_dim, 1)

        # creating optimizer and criterion
        self.optimizer, self.criterion = self.__compile(criterion, optimizer, learning_rate)

    def __compile(self, criterion, optimizer, learning_rate):

        optimizer = get_pytorch_optimizer(optimizer, self.parameters(), learning_rate)
        criterion = get_pytorch_criterion(criterion)

        return optimizer, criterion

    def forward(self, X):

        y = self.linear_layer(X)

        return y

    def fit(self, X, y):

        # resetting gradients w.r.t. weights
        self.optimizer.zero_grad()

        # passing input forward to get outputs
        y_ = self.forward(X)

        # calculating loss + getting gradients
        loss = self.criterion(y_, y)
        loss.backward()

        # updating weights
        self.optimizer.step()

        # calculating training accuracy
        return loss.item()

    def predict(self, X):

        # passing input forward to get outputs
        y_ = self.forward(X)

        return y_
