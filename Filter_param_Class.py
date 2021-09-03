
import numpy as np
import torch
import torch.fft
from torch import nn
from Individual_Class import Individual
import matplotlib.pyplot as plt



class Filter_param(nn.Module):

    def __init__(self, filtro_in):
        super(Filter_param, self).__init__()

        self.name = 'Filter_param'

        self.limits = filtro_in.parameters

        # for each parameter gets the normalized parameter value saved in the individual passed and saves it as an attribute of the filter class.
        for param in self.limits:
            value = getattr(filtro_in,param + '_normal')
            setattr(self,param + '_normal',nn.Parameter(torch.tensor(value)))

    def forward(self, X, F):

        self.c = ((2 ** (1 / 3) - 1) ** (1 / 2)) * F / np.pi


        t = torch.arange(X.shape[1]) / X.shape[1]


        # it takes the normalized value of the parameter that was saved (either from initialization or from previous time steps) and computes the non-normalized value.
        for param in self.limits:
            value = getattr(self,param+'_normal')
            low = self.limits[param][0]
            high = self.limits[param][1]
            setattr(self,param,low + (high-low)*torch.sigmoid(value))


        self.f = self.f_inf + (self.f_start - self.f_inf) * (0.7 * 0.1**self.f_decay) ** (t / (10*0.1**self.f_T))
        self.w = torch.sigmoid((t - self.w_offset) / self.w_T)


        self.a = (self.f.view(1,-1) - self.c) / (self.f.view(1,-1) + self.c)
        self.b = (self.a + 1) / 2


        self.y_0 = []
        self.y_1 = []
        self.y_2 = []

        self.y_plot = torch.zeros(X.shape[0], X.shape[1])

        self.averaged_summed = 0

        for t in range(X.shape[1]):
            self.y_0.append(X[:,t].detach().clone().requires_grad_(True))

            if t == 0:

                self.y_1.append(X[:, t].detach().clone().requires_grad_(True))
                self.y_2.append(X[:, t].detach().clone().requires_grad_(True))
            else:

                self.y_1.append(self.b[:,t] * (self.y_0[-1] + self.y_0[-2]) - self.a[:,t] * self.y_1[-1])
                self.y_2.append(self.b[:,t] * (self.y_1[-1] + self.y_1[-2]) - self.a[:,t] * self.y_2[-1])

            self.y_plot[:, t] = self.y_2[-1].detach()

            self.averaged_summed += self.w[t] * self.y_2[-1]

        normalization = torch.sum(self.w)

        self.prediction = self.averaged_summed / normalization


if __name__ == '__main__':
    print(torch.__version__)
    params = {'f_inf': [0.01, 10], 'f_start': [10, 120], 'f_decay': [0, 3], 'f_T': [0,4],
              'w_T': [0.001, 1], 'w_offset': [0, 1.]}

    x = np.random.normal(0,1,size = (len(params.keys())))

    Individual = Individual(params,x,L = 100)
    a = Filter_param(Individual)
    X = torch.rand(5,1000)/1000 + 5 + torch.sin(torch.arange(1000)*0.1).reshape(-1,1000)
    F = torch.ones(5,1) * 100
    #F = 100
    a.forward(X,F)

    print(a.f.shape)
    plt.plot(a.y_plot[0])
    plt.show()
    plt.plot(a.f.detach())
    plt.show()