import torch
import torch.nn as nn
from torch.autograd import Variable

from utils.nn_func import *
from utils.nn_dict import *


class LogReg(nn.Module):
    """
    Logistic Regression Class
    """

    def __init__(self, input_dim, output_dim,
                 criterion=LossDict.cross_entropy, optimizer=OptimizerDict.sgd,
                 learning_rate=0.01):

        # calling the super class
        super(LogReg, self).__init__()

        # initializing the linear layer
        self.linear_layer = nn.Linear(input_dim, output_dim)

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

        # passing the input forward to get the outputs
        y_ = self.forward(X)

        # calculating loss + getting gradients
        loss = self.criterion(y_, y)
        loss.backward()

        # updating weights
        self.optimizer.step()

        # calculating training loss
        return loss.item()

    def evaluate(self, X, y):

        # passing input forward to get outputs
        y_ = self.forward(X)

        # calculating loss + getting gradients
        loss = self.criterion(y_, y)

        # calculating accuracy
        _, y_ = y_.data.max(1, keepdim=False)
        acc = (y_ == y).cpu().sum().item() / len(y)

        # returning loss and accuracy
        return loss.item(), acc

    def predict(self, X):

        # passing the input forward to get the outputs
        y_ = self.forward(X)
        _, y_ = y_.data.max(1, keepdim=False)

        return y_
