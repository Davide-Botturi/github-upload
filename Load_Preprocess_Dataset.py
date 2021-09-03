import pandas as pd
from Resample_Dataset import resample_constant_length
from Compute_spectral_centroid import spectral_centroid
import torch
import numpy as np
import matplotlib.pyplot as plt


def load_dataset(path):
    df = pd.read_pickle(path)
    return df

def same_length_resampling(df,length,F_s):
    signal = []
    sample_frequency = []
    for i in df['loadcell']:
        tmp_signal,tmp_frequency_sampling = resample_constant_length(i,F_s,length)
        signal.append(tmp_signal)
        sample_frequency.append(tmp_frequency_sampling)

    df["loadcell_" + str(length)] = signal
    df["frequency_sampling_" + str(length)] = sample_frequency

    return df

def compute_spectral_centroid(df,length):
    features = []

    for j,i in enumerate(df['loadcell_' + str(length)]):
        sample_rate = df['frequency_sampling_' + str(length)][j]


        n_fft = length//3
        hop_length = n_fft

        tmp_features = spectral_centroid(torch.tensor(i), sample_rate=sample_rate, pad=0,
                          window=torch.hann_window(n_fft), n_fft=n_fft, hop_length=hop_length, win_length=n_fft,
                          power=1)
        features.append(tmp_features)
    df["features_"+ str(length)] = features
    return df

def add_mass_velocity_label(df):
    mass_velocity_label = []
    for i in range(df.shape[0]):
        mass_velocity_label.append(df['nominal_label'].iloc[i] +'_'+ df['v'].iloc[i])
    df['mass_velocity_label'] = mass_velocity_label
    return df


def create_train_validation_datasets(df,train_labels,eval_labels,length):
    df = df.sample(frac=1).reset_index(drop=True)
    X_train, X_eval, y_train, y_eval, features_train, features_eval , labels_train, labels_eval, sample_freq_train,\
    sample_freq_eval, v_train, v_eval =\
        [[] for i in range(12)]

    for i in range(df.shape[0]):
        #label_mode = 'nominal_label'       #with this option the same mass at 2 different velocities is shown in the same histogram
        label_mode = 'mass_velocity_label'  # with this option the masses are divided by mass AND velocity in the histogram showing the accuracy


        if df['nominal_label'].iloc[i] in train_labels:
            X_train.append(df['loadcell_' + str(length)].iloc[i])
            y_train.append(df['label'].iloc[i])
            features_train.append(np.array(df['features_' + str(length)].iloc[i])) #Converted to numpy cause it was a torch tensor
            labels_train.append(df[label_mode].iloc[i])
            sample_freq_train.append(df['frequency_sampling_' + str(length)].iloc[i])
            v_train.append(df['v'].iloc[i])



        elif df['nominal_label'].iloc[i] in eval_labels:
            X_eval.append(df['loadcell_' + str(length)].iloc[i])
            y_eval.append(df['label'].iloc[i])
            features_eval.append(np.array(df['features_' + str(length)].iloc[i])) #Converted to numpy cause it was a torch tensor
            labels_eval.append(df[label_mode].iloc[i])
            sample_freq_eval.append(df['frequency_sampling_' + str(length)].iloc[i])
            v_eval.append(df['v'].iloc[i])

        else:
            print("This Label is neither in train or in evaluation!",df['nominal_label'].iloc[i])

    return np.array(X_train),np.array(X_eval),np.array(y_train),np.array(y_eval),np.array(features_train),np.array(features_eval),\
           np.array(labels_train),np.array(labels_eval),np.array(sample_freq_train),np.array(sample_freq_eval),np.array(v_train),np.array(v_eval)


if __name__ == '__main__':
    # keys ['loadcell', 'FC1', 'FC2', 'label', 'nominal_label', 'v']
    #{'1122', '369', '1123', '263', '373', '371', '264', '361', '81', '265', '902', '250', '557', '56', '121', '900', '899', '376', '83', '179', '82', '33', '84', '247', '248', '905', '246'}
    path = '/home/davide/Desktop/EnneGi/Prove_SAR_Matteo_Caraffini/Prove/DataFrame.pkl'

    df = load_dataset(path)
    print(df.keys())

    F_s = 50000 #Hz
    length = 800

    df = same_length_resampling(df,length,F_s)
    df = compute_spectral_centroid(df,length)

    #PLOTS TO SEE THE RESAMPLING
    l = len(df['loadcell'][20])
    plt.plot(np.arange(l)/l,df['loadcell'][20])
    plt.plot(np.arange(length)/length, df['loadcell_'+ str(length)][20])

    print("Column of the new sampling frequency",df['frequency_sampling_' + str(length)])

    #print(df['features_'+ str(length)].values)
    #print(df['features_' + str(length)][1].shape)

    train_labels = ['1122', '369', '1123', '263', '373', '371', '264', '361', '81', '265', '902', '250', '557', '56', '121', '900', '899', '376', '83', '179']
    eval_labels = ['82', '33', '84', '247', '248', '905', '246']

    X_train, X_eval, y_train, y_eval, features_train, features_eval, labels_train, labels_eval, sample_freq_train,sample_freq_eval, v_train, v_eval = \
        create_train_validation_datasets(df,train_labels,eval_labels,length)

    print("Check the shapes of the given dataset",X_train.shape,X_eval.shape,y_train.shape,features_eval.shape,sample_freq_train.shape)
    print(v_eval)

