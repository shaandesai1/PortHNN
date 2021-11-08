import torch.nn as nn
import torch
import numpy as np
from .baseline import *
class HNN(base_model):
    def __init__(self, input_dim, hidden_dim, output_size, deltat):
        super(HNN, self).__init__(input_dim,hidden_dim,output_size,deltat)

    def time_deriv(self, x,t):
        input_vec = x
        F2 = self.get_H(input_vec)
        dF2 = torch.autograd.grad(F2.sum(), input_vec, create_graph=True)[0]
        return dF2@self.M.t()

