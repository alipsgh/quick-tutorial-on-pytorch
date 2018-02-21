
import torch
from torch.autograd import Variable


class RBM:

    def __init__(self, vis_dim, hid_dim, k, learning_rate=0.1, use_cuda=True):

        self.W = torch.randn(vis_dim, hid_dim) * 0.0001
        self.v_bias = torch.zeros(vis_dim)
        self.h_bias = torch.zeros(hid_dim)

        self.k = k
        self.learning_rate = learning_rate
        self.use_cuda = use_cuda

        if torch.cuda.is_available() and self.use_cuda:
            self.W = self.W.cuda()
            self.v_bias = self.v_bias.cuda()
            self.h_bias = self.h_bias.cuda()

    def sample_h_given_v(self, v_s):
        h_p = torch.sigmoid(torch.mm(v_s, self.W) + self.h_bias)
        h_s = torch.bernoulli(h_p)
        return [h_p, h_s]

    def sample_v_given_h(self, h_s):
        v_p = torch.sigmoid(torch.mm(h_s, self.W.t()) + self.v_bias)
        v_s = torch.bernoulli(v_p)
        return [v_p, v_s]

    def gibbs_hvh(self, h_s):
        v_p, v_s = self.sample_v_given_h(h_s)
        h_p, h_s = self.sample_h_given_v(v_s)
        return [v_p, v_s, h_p, h_s]

    def gibbs_vhv(self, v_s):
        h_p, h_s = self.sample_h_given_v(v_s)
        v_p, v_s = self.sample_v_given_h(h_s)
        return [h_p, h_s, v_p, v_s]

    def free_energy(self, v):
        v_bias_term = torch.mv(v, self.v_bias)
        wx_b = torch.mm(v, self.W) + self.h_bias
        hidden_term = torch.sum(torch.log(1 + torch.exp(wx_b)), dim=1)
        return -v_bias_term - hidden_term

    def fit(self, x):

        if torch.cuda.is_available() and self.use_cuda:
            x = x.cuda()
        v_s = x

        # calculate positive part :: 'p' stands for positive
        ph_p, ph_s = self.sample_h_given_v(v_s)

        # calculate negative part :: 'n' stands for negative
        nv_p, nv_s, nh_p, nh_s = None, None, None, ph_s
        for _ in range(self.k):
            nv_p, nv_s, nh_p, nh_s = self.gibbs_hvh(nh_s)

        # calculate loss
        cost = torch.mean(self.free_energy(v_s)) - torch.mean(self.free_energy(nv_s))

        # update parameters
        batch_size = v_s.shape[0]
        self.W = self.W + self.learning_rate * (torch.mm(v_s.t(), ph_p) - torch.mm(nv_s.t(), nh_p)) / batch_size
        self.v_bias = self.v_bias + self.learning_rate * (v_s - nv_s).sum(0) / batch_size
        self.h_bias = self.h_bias + self.learning_rate * (ph_s - nh_p).sum(0) / batch_size

        # calculate
        loss = self.cal_cross_entropy(v_s, nv_p)

        return cost, loss

    @staticmethod
    def cal_cross_entropy(p, p_):
        return torch.mean(torch.sum(p * torch.log(p_) + (1 - p) * torch.log(1 - p_), dim=1))

    def compress(self, x):

        if torch.cuda.is_available():
            x = x.cuda()
        v = x

        h_p, _ = self.sample_h_given_v(v)

        return h_p

    def reconstruct(self, x):

        if torch.cuda.is_available() and self.use_cuda:
            x = x.cuda()
        v_s = x

        h_p, h_s = self.sample_h_given_v(v_s)
        v_p, v_s = self.sample_v_given_h(h_s)

        return v_s
