import numpy as np
import torch
import torch.fft
from torch import nn
from Individual_Class import Individual
import matplotlib.pyplot as plt
from Compute_spectral_centroid import spectral_centroid

#I think I should optimize employing filter_param which is the baseline and then compute the correction based on signal features.
#filtro_in.parameters contains the max and min value that the filter parameter can achieve. It's a dictionary 'limit_name':[min_vaue,max_value]

class Filter_feature(nn.Module):

    def __init__(self, filtro_in , n_features, filter_name = 'Filter_feature',n_hidden = 8):
        super(Filter_feature, self).__init__()

        self.name = filter_name


        self.limits = filtro_in.parameters

        self.n_features = n_features
        self.n_hidden = n_hidden

        self.N_param = len(self.limits)


        self.layers = nn.Sequential(
            #torch.nn.MaxPool1d(3),
            torch.nn.BatchNorm1d(self.n_features),
            torch.nn.Linear(self.n_features, self.n_hidden),
            #torch.nn.Dropout(p = 0.2),
            torch.nn.BatchNorm1d(self.n_hidden),
            torch.nn.ReLU(),

            torch.nn.Linear(self.n_hidden, self.n_hidden), # To return to previous configuration comment this line
            torch.nn.BatchNorm1d(self.n_hidden), #and this
            torch.nn.ReLU(), # and this

            torch.nn.Linear(self.n_hidden, self.N_param),
            #torch.nn.Dropout(p = 0.2)
        )

        # for each parameter gets the normalized parameter value saved in the individual passed and saves it as an attribute of the filter class.
        for param in self.limits:
            value = getattr(filtro_in,param + '_normal')
            setattr(self,param + '_normal',nn.Parameter(torch.tensor(value)))

    def forward(self, X, features, F):

        t = torch.arange(X.shape[1]).repeat(X.shape[0],1) / X.shape[1]
        self.c = ((2 ** (1 / 3) - 1) ** (1 / 2)) * F / np.pi

        # Features enter into the sequential NN which outputs the correction for each parameter.
        self.feature_correction = self.layers(features)



        # takes each parameter value(comes from the initialization, it's the same for all the samples) and corrects it with the correction computed
        # by sequential NN (correction is different for each sample)

        for j,param in enumerate(self.limits):
            value = getattr(self,param+'_normal')
            feature_correction = self.feature_correction[:,j]
            low = self.limits[param][0]
            high = self.limits[param][1]
            setattr(self,param,low + (high-low)*torch.sigmoid(feature_correction + value).reshape(-1,1))


        self.f = self.f_inf + (self.f_start - self.f_inf) * (0.7 * 0.1**self.f_decay) ** (t / (10*0.1**self.f_T))
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
    params = {'f_inf': [0.01, 10], 'f_start': [10, 120], 'f_decay': [0, 3], 'f_T': [0,4],
              'w_T': [0.001, 1], 'w_offset': [0, 1.]}


    sample_rate = 5000

    x = np.random.normal(0,1,size = (len(params.keys())))
    Individual = Individual(params,x,L = 100)


    t = np.arange(5000)/5000 # 1 secondo
    f1= 55 #Hz
    f2 = 40


    X1 = np.sin(2*np.pi*f1*t)
    X2 = 5 * np.sin(2*np.pi*f2 * t)

    X1 = torch.tensor(X1).view(1,-1)
    X2 = torch.tensor(X2).view(1, -1)


    X = torch.cat((X1,X2),dim = 0)


    n_fft = 1000
    hop_length = 500

    feature1 = spectral_centroid(X1,sample_rate= sample_rate ,pad = 0,
                            window = torch.hann_window(n_fft),n_fft = n_fft, hop_length= hop_length , win_length= n_fft, power = 1)

    feature2 = spectral_centroid(X2,sample_rate= sample_rate ,pad = 0,
                            window = torch.hann_window(n_fft),n_fft = n_fft, hop_length= hop_length , win_length= n_fft, power = 1)


    feature1 = (feature1).view(1,-1)
    feature2 = (feature2).view(1, -1)

    features = torch.cat((feature1,feature2),dim = 0)

    F = torch.ones(2,1)*sample_rate

    print(features.dtype,X.dtype)

    a = Filter_feature(Individual,n_features= features.shape[1])

    a.forward(X.float(),features.float(),F)

    print(a.f.shape)
    plt.plot(a.y_plot[0])
    plt.show()
    print(a.f.shape)

    plt.plot(np.array(a.f.detach()).T)

    plt.show()

