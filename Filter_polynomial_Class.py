import numpy as np
import torch
import torch.fft
from torch import nn
from Individual_Class import Individual
import matplotlib.pyplot as plt


#I think I should optimize employing filter_param which is the baseline and then compute the correction based on poly(or other) signal features.

class Filter_poly(nn.Module):

    def __init__(self, filtro_in , F = 50000):
        super(Filter_poly, self).__init__()

        self.name = 'Filter_polynomial'
        self.F = F
        self.c = ((2 ** (1 / 3) - 1) ** (1 / 2)) * self.F / np.pi
        self.limits = filtro_in.parameters


        self.N_param = len(self.limits)
        self.N_polynomial = 5


        self.layers = nn.Sequential(
            #torch.nn.MaxPool1d(3),
            torch.nn.BatchNorm1d(self.N_polynomial + 1),
            torch.nn.Linear(self.N_polynomial + 1, 8),
            #torch.nn.Dropout(p = 0.2),
            torch.nn.BatchNorm1d(8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, self.N_param),
            #torch.nn.Dropout(p = 0.2)
        )

        # for each parameter gets the normalized parameter value saved in the individual passed and saves it as an attribute of the filter class.
        for param in self.limits:
            value = getattr(filtro_in,param + '_normal')
            setattr(self,param + '_normal',nn.Parameter(torch.tensor(value)))

    def forward(self, X):

        t = torch.arange(X.shape[1]).repeat(X.shape[0],1) / X.shape[1]

        #fits J polynomials of order self.N_polynomial (J is the number of examples in the "batch")
        #ATTENTIONNNN CHECK IF THE POLY FEATURES ARE DESCRIPTIVE AND DON'T CHANGE
        y = np.array(X.detach())
        x = np.arange(y.shape[1])/y.shape[1]

        poly_features = torch.Tensor(np.polyfit(x, y.reshape(x.shape[0],-1), self.N_polynomial).T)

        # Poly features enter into the sequential NN which outputs the correction for each parameter.
        self.poly_correction = self.layers(poly_features)


        for j,param in enumerate(self.limits):
            value = getattr(self,param+'_normal')
            poly_correction = self.poly_correction[:,j]
            low = self.limits[param][0]
            high = self.limits[param][1]
            setattr(self,param,low + (high-low)*torch.sigmoid(poly_correction + value).reshape(-1,1))

        self.f = self.f_inf + (self.f_start - self.f_inf) * self.f_decay ** (t / self.f_T)
        self.w = torch.sigmoid((t - self.w_offset) / self.w_T)

        self.a = (self.f - self.c) / (self.f + self.c)
        self.b = (self.a + 1) / 2


        self.y_0 = []
        self.y_1 = []
        self.y_2 = []

        self.y_plot = torch.zeros(X.shape[0], X.shape[1])

        self.averaged_summed = 0

        for t in range(X.shape[1]):
            self.y_0.append(X[:, t].detach().clone().requires_grad_(True))
            #t_ = torch.tensor(t)

            if t == 0:

                self.y_1.append(X[:, t].detach().clone().requires_grad_(True))
                self.y_2.append(X[:, t].detach().clone().requires_grad_(True))
            else:

                self.y_1.append(self.b[:,t] * (self.y_0[-1] + self.y_0[-2]) - self.a[:,t] * self.y_1[-1])
                self.y_2.append(self.b[:,t] * (self.y_1[-1] + self.y_1[-2]) - self.a[:,t] * self.y_2[-1])

            self.y_plot[:, t] = self.y_2[-1].detach()

            self.averaged_summed += self.w[:,t] * self.y_2[-1]

        normalization = torch.sum(self.w,dim = 1)

        self.prediction = self.averaged_summed / normalization



if __name__ == '__main__':
    print(torch.__version__)
    params = {'f_inf': [0.01, 10], 'f_start': [10, 120], 'f_decay': [0.001, 0.5], 'f_T': [0.0001, 1],
              'w_T': [0.001, 1], 'w_offset': [0, 1.]}

    x = np.random.normal(0,1,size = (len(params.keys())))
    Individual = Individual(params,x,L = 100,F = 1000)
    a = Filter_poly(Individual,F = 1000)

    a.forward((torch.rand(5,1000)/1000) + 5 + torch.sin(torch.arange(1000)*0.1).reshape(-1,1000))

    print(a.f.shape)
    plt.plot(a.y_plot[0])
    plt.show()
    print(a.f.shape)

    plt.plot(np.array(a.f.detach()).T)

    plt.show()

