import pandas as pd
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

def resample_constant_length(data,original_sample_frequency,length):
    t = np.arange(length)/length
    f = interpolate.interp1d(np.arange(len(data))/len(data),data)
    sample_ratio = len(data)/length #Ratio between original sample length and actual length --> same ratio of sample frequency
    return f(t),original_sample_frequency/sample_ratio


if __name__ == '__main__':
    t = np.arange(10000)
    a = np.sin(t/1000)
    F_s = 50000 # Hz

    length = 37
    plt.plot(resample_constant_length(a,F_s,length))
    plt.plot(np.arange(len(a))/len(a) * length,a)
    plt.show()