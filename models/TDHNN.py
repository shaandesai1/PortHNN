import torch.nn as nn
import torch
from .baseline import *

class TDHNN(base_model):
    def __init__(self, input_dim, hidden_dim, output_size, deltat):
        super(TDHNN, self).__init__(input_dim,hidden_dim,output_size,deltat)
        self.input_dim = input_dim - 1
    def time_deriv(self, x, t):
        input_vec = torch.cat([x,t.reshape(-1,1)],1)
        H = self.get_H(input_vec)
        dF2 = torch.autograd.grad(H.sum(), x, create_graph=True)[0]
        return dF2@self.M.t()












