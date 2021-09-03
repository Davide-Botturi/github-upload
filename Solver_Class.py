import numpy as np
import torch
#import torch.fft
from torch import nn



class Solver():
    def __init__(self, Filter, lr=0.01):
        self.lr = lr
        self.model = Filter
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = torch.nn.MSELoss()

        self.losses = []
        self.eval_losses = []