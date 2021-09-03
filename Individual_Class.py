import numpy as np

# parameters is a dict where is specified the max and minimum values that the parameters of the filter can have
# The value of the i-th parameter of the individual is computed as (max-min)*sigmoid(x[i]) + min
# L is the length of the signal, F is the sampling rate
# x is the array that specifies which normalized value will have each parameter. (Normalized just means that can take any value and will be later corrected through the limits each parameter has)

class Individual():
    def __init__(self, parameters, x, L):
        self.parameters = parameters


        for i, param in enumerate(self.parameters):
            setattr(self, param,
                    self.parameters[param][0] + (self.parameters[param][1] - self.parameters[param][0]) * self.sigmoid(
                        x[i]))
            setattr(self, param + '_normal', x[i])

        self.t = np.arange(0, L) / L


        self.f = self.f_inf + (self.f_start - self.f_inf) * (0.7 * 0.1**self.f_decay) ** (self.t / (10*0.1**self.f_T))
        self.w = self.sigmoid((self.t - self.w_offset) / self.w_T)



    def sigmoid(self, x):
        return np.exp(x) / (1 + np.exp(x))

    def init_from_trained_filter(self,filter):
        for param in self.parameters:
            value_normal = getattr(filter,param + '_normal').item()
            value = getattr(filter,param).item()

            setattr(self,param + '_normal',value_normal)
            setattr(self, param, value)

        self.f = self.f_inf + (self.f_start - self.f_inf) * (0.7 * 0.1**self.f_decay) ** (self.t / (10*0.1**self.f_T))
        self.w = self.sigmoid((self.t - self.w_offset) / self.w_T)


if __name__ == '__main__':
    params = {'f_inf': [0.01, 10], 'f_start': [10, 120], 'f_decay': [0.001, 0.5], 'f_T': [0.0001, 1],
              'w_T': [0.001, 1], 'w_offset': [0, 1.]}

    x = np.random.normal(0, 1, size=(len(params.keys())))
    Individual = Individual(params, x, 100)
    print(Individual.__dict__)