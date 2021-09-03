import numpy as np 
import scipy.io.wavfile

t = np.arange(45000)/5000 
a = np.random.rand(10,45000)/50 + np.sin(t) + np.cos(50*t) + np.sin(0.01*t + 0.1)
a[:,25000:] += np.sin(500*t[25000:])

for i,j in enumerate(a):
    scipy.io.wavfile.write(str(i)+'.wav',4500,j)
