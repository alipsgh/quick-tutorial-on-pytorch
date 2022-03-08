import torch
import torch.nn as nn
from torch.autograd import Variable
from utils.funcs import get_pytorch_optimizer, get_pytorch_criterion, get_pytorch_activation


class ANN(nn.Module):
    """
    Class of Artificial Neural Network (ANN)
    """

    def __init__(self, input_dim, hidden_layers_dim, output_dim, activations,
                 criterion='CrossEntropy', optimizer='sgd', learning_rate=0.01):

        super(ANN, self).__init__()

        # initializing lists of linear and activation layers
        self.linear_layers = nn.ModuleList()
        self.activation_layers = nn.ModuleList()

        # adding linear and output layers
        for i, hidden_dim in enumerate(hidden_layers_dim):
            if i == 0:
                self.linear_layers.append(nn.Linear(input_dim, hidden_dim))
            else:
                self.linear_layers.append(nn.Linear(self.linear_layers[-1].out_features, hidden_dim))
            self.activation_layers.append(get_pytorch_activation(activations[i]))
        self.output = nn.Linear(self.linear_layers[-1].out_features, output_dim)

        # creating optimizer and criterion
        self.optimizer, self.criterion = self.__compile(criterion, optimizer, learning_rate)

        if torch.cuda.is_available():
            self.cuda()

    def __compile(self, criterion, optimizer, learning_rate):
        optimizer = get_pytorch_optimizer(optimizer, self.parameters(), learning_rate)
        criterion = get_pytorch_criterion(criterion)
        return optimizer, criterion

    def forward(self, X):
        for linear, activation in zip(self.linear_layers, self.activation_layers):
            X = activation(linear(X))
        y = self.output(X)
        return y

    def fit(self, X, y):

        if torch.cuda.is_available():
            X, y = X.cuda(), y.cuda()
        X, y = Variable(X), Variable(y)

        # resetting gradients w.r.t. weights
        self.optimizer.zero_grad()

        # passing input forward to get outputs
        y_ = self.__call__(X)

        # calculating loss + getting gradients
        loss = self.criterion(y_, y)
        loss.backward()

        # updating weights
        self.optimizer.step()

        # calculating training accuracy
        return loss.data[0]

    def predict(self, X):

        if torch.cuda.is_available():
            X = X.cuda()
        X = Variable(X)

        # passing input forward to get outputs
        y_ = self.__call__(X)
        _, y_ = y_.data.max(1, keepdim=False)

        if torch.cuda.is_available():
            y_ = y_.cpu()

        return y_
