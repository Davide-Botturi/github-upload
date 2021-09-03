import pandas as pd
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
import torchaudio
import torch

# Input waveform (torch tensor) and gives back spectral centroids of the stft of the waveform. n_fft is the number of sample taken into each fft window.
#hop length is the distance between the start of one window and the start of the next.



def spectral_centroid(
        waveform,
        sample_rate,
        pad,
        window,
        n_fft,
        hop_length,
        win_length,
        power) :
    specgram = torchaudio.functional.spectrogram(waveform, pad=pad, window=window, n_fft=n_fft, hop_length=hop_length,
                           win_length=win_length, power=power, normalized=False,center = False)

    #treshold = torch.max(specgram) *10 **-3
    #specgram[torch.where(specgram < treshold)] = 0
    #specgram[40:,:] = 0
    #specgram[0,:] = 0

    freqs = torch.linspace(0, sample_rate // 2, steps=1 + n_fft // 2,
                           device=specgram.device).reshape((-1, 1))
    freq_dim = -2
    return (freqs * specgram).sum(dim=freq_dim) / specgram.sum(dim=freq_dim)


if __name__ == '__main__':
    t = np.arange(50000)/50000 # 1 secondo
    f1= 55 #Hz
    f2 = 40
    f3 = 20
    a = np.sin(2*np.pi*f1*t) + 5 * np.sin(2*np.pi*f2 * t) + 2 * np.cos(2*np.pi*f3 * t)

    #plt.plot(a)
    #plt.show()

    sample_rate = 50000
    n_fft = 5000
    hop_length = 2500

    print(spectral_centroid(torch.tensor(a),sample_rate= sample_rate ,pad = 0,
                            window = torch.hann_window(n_fft),n_fft = n_fft, hop_length= hop_length , win_length= n_fft, power = 1))

    print((f1 + 5*f2+f3*2)/8)

    #plt.plot(np.abs(np.fft.rfft(a)))
    #plt.show()