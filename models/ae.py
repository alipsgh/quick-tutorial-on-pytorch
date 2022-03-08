
import torch
import torch.nn as nn

from torch.autograd import Variable
from utils.funcs import get_pytorch_activation, get_pytorch_optimizer, get_pytorch_criterion
from utils.dicts import *


class AE(nn.Module):

    def __init__(self, input_dim, encoders_dim, encoder_activations,
                 criterion=LossDict.binary_cross_entropy, optimizer='adam', learning_rate=0.001):

        super(AE, self).__init__()

        self.linear_layers = nn.ModuleList()
        self.activation_layers = nn.ModuleList()

        # adding encoders
        for i, encoder_dim in enumerate(encoders_dim):
            if i == 0:
                self.linear_layers.append(nn.Linear(input_dim, encoder_dim))
            else:
                self.linear_layers.append(nn.Linear(self.linear_layers[-1].out_features, encoder_dim))
            self.activation_layers.append(get_pytorch_activation(encoder_activations[i]))

        # adding decoders
        decoders_dim = list(reversed(encoders_dim[:-1]))
        decoders_activations = list(reversed(encoder_activations[:-1]))
        for i, decoder_dim in enumerate(decoders_dim):
            self.linear_layers.append(nn.Linear(self.linear_layers[-1].out_features, decoder_dim))
            self.activation_layers.append(get_pytorch_activation(decoders_activations[i]))

        self.activation_layers.append(nn.Sigmoid())
        self.linear_layers.append(nn.Linear(self.linear_layers[-1].out_features, input_dim))

        # creating optimizer and criterion
        self.optimizer, self.criterion = self.__compile(criterion, optimizer, learning_rate)

        if torch.cuda.is_available():
            self.cuda()

    def __compile(self, criterion, optimizer, learning_rate):
        optimizer = get_pytorch_optimizer(optimizer, self.parameters(), learning_rate)
        criterion = get_pytorch_criterion(criterion)
        return optimizer, criterion

    def forward(self, x):
        for i in range(len(self.activation_layers)):
            x = self.activation_layers[i](self.linear_layers[i](x))
        # x_ = self.linear_layers[-1](x)
        x_ = x
        return x_

    def fit(self, x):

        if torch.cuda.is_available():
            x = x.cuda()
        x = Variable(x)

        # resetting gradients w.r.t. weights
        self.optimizer.zero_grad()

        # passing input forward to get outputs
        x_ = self.__call__(x)

        # calculating loss + getting gradients
        loss = self.criterion(x_, x)
        loss.backward()

        # updating weights
        self.optimizer.step()

        # calculating training accuracy
        return loss.item()

    def reconstruct(self, x):

        if torch.cuda.is_available():
            x = x.cuda()
        x = Variable(x)

        x_ = self.__call__(x)

        return x_

    def encode(self, x):

        if torch.cuda.is_available():
            x = x.cuda()
        x_c = Variable(x)

        # let's encode X and get its compressed value, i.e. x_c
        c_index = int(len(self.linear_layers) / 2) - 1
        for i, (linear, activation) in enumerate(zip(self.linear_layers, self.activation_layers)):
            x_c = activation(linear(x_c))
            if i == c_index:
                break

        if torch.cuda.is_available():
            x_c = x_c.cpu()

        return x_c

    def get_weight(self):

        Ws = []
        Wm = None
        c_index = int(len(self.linear_layers) / 2) - 1
        for i, linear in enumerate(self.linear_layers):
            w = linear.weight.t()
            Ws.append(w)
            if i == 0:
                Wm = w
            else:
                Wm = torch.mm(Wm, w)
            if i == c_index:
                break

        return Ws, Wm




