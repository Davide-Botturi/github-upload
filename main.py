import numpy as np
import torch
import matplotlib.pyplot as plt
import pickle
import os

#STEP 1
from Load_Preprocess_Dataset import same_length_resampling,compute_spectral_centroid,create_train_validation_datasets,load_dataset,add_mass_velocity_label
#STEP 2
from Individual_Class import Individual
from Filter_param_Class import Filter_param
from Solver_Class import Solver

#Step 3
from Train_Solver import train_solver,plot_loss
from Visualize_Accuracy import graph_accuracy


#Step 4
from Filter_generic_feature import Filter_feature


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    '''
    STEP 1: DATASET CREATION
    -Loads the dataset
    -Resamples all the data to have the same length
    -Computes the features vector(in this case spectral centroid)
    -Inspects the dataset and separates between train and eval data
    - Converts into Tensors the dataset and converts mV/V in grams

    '''
    print("STEP 1.....DATASET CREATION")
    path = '/home/davide/Desktop/EnneGi/Prove_SAR_Matteo_Caraffini/Prove/DataFrame.pkl'
    experiment_path = "/home/davide/Desktop/EnneGi/Experiments_Prove_Sar/Experiment_10_bigger_network_long_epochs_low_lr/"

    if ('Initialization' not in os.listdir(experiment_path)):
        os.mkdir(experiment_path + 'General')
    os.mkdir(experiment_path + 'Feature')

    df = load_dataset(path)
    l = df[np.logical_and(df['v'] == 'v1', df['nominal_label'] == '33')]['label'].index
    df.loc[l,'label'] = 0.0006598544000000001 #correction for probably wrong label.

    F_s = 50000  # Sample frequency of the acquired data
    length = 800 # Length of the resampled signals

    df = same_length_resampling(df, length, F_s)
    df = compute_spectral_centroid(df, length)
    df = add_mass_velocity_label(df)

    train_labels = ['1122', '369', '1123', '263', '373', '371', '264', '361', '81', '265', '902', '250', '557', '56',
                    '121', '900', '899', '376', '83', '179']
    eval_labels = ['82', '33', '84', '247', '248', '905', '246']

    X_train, X_eval, y_train, y_eval, features_train, features_eval, labels_train, labels_eval, sample_freq_train, sample_freq_eval, v_train, v_eval = \
        create_train_validation_datasets(df, train_labels, eval_labels, length)

    sensitivity = 6.72*10**-8 # mV/(V*g)

    X_train = torch.tensor(X_train) / sensitivity
    y_train = torch.tensor(y_train) / sensitivity
    X_eval = torch.tensor(X_eval) / sensitivity
    y_eval = torch.tensor(y_eval) / sensitivity
    sample_freq_train = torch.tensor(sample_freq_train).view(-1,1)
    sample_freq_eval = torch.tensor(sample_freq_eval).view(-1,1)
    features_train = torch.tensor(features_train)
    features_eval = torch.tensor(features_eval)

    '''
    STEP 2: GENERIC FILTER CREATION WITH INITIALIZATION
    - Defines the max and min values that the parameters of the filter can acquire
    - Initialize (an Individual and) the Filter, it contains the parameters of the filter used for initialization. The zero initialization is equivalent of choosing
    (min + max)/2
    - Creates the Solver (used to train the filter)

    '''
    print("STEP 2..... GENERIC FILTER CREATION WITH INITIALIZATION")

    params = {'f_inf': [0.01, 10], 'f_start': [10, 120], 'f_decay': [0, 3], 'f_T': [0,4],
              'w_T': [0.001, 1], 'w_offset': [0, 1.]}

    lr = 0.01
    first_individual = Individual(params, np.zeros(len(params.keys())), length)
    first_filter = Filter_param(first_individual)
    solver = Solver(first_filter,lr = lr)

    plt.plot(first_individual.f)
    plt.title('Frequency cut_off initialization filter')
    plt.savefig(experiment_path +'Frequency cut_off initialization filter')
    plt.clf()
    plt.close()

    '''
    STEP 3
    - Trains the generic filter
    - Save graphs of results in the experiment dir
    '''
    print("STEP 3..... GENERIC FILTER OPTIMIZATION")

    n_epochs = 2000

    if 'General_filter' in os.listdir(experiment_path):
        with open(experiment_path + 'General_filter', 'rb') as Optimized_Filter_Param:
            first_individual = pickle.load(Optimized_Filter_Param)
        print("READING PREEXISTING FILTER")
    else:
        train_predictions, eval_predictions = train_solver(solver,X_train,sample_freq_train,y_train,X_eval,sample_freq_eval,y_eval, n_epochs)

        # GRAPH OPTIMIZATION RESULTS

        graph_accuracy(np.array(y_train), np.array(train_predictions), v_train,experiment_path + "General/Train_accuracy_v_labels.png")
        graph_accuracy(np.array(y_eval), np.array(eval_predictions), v_eval,experiment_path + "General/Eval_accuracy_v_labels.png")

        graph_accuracy(np.array(y_train), np.array(train_predictions), labels_train,experiment_path + "General/Train_accuracy_mass_labels.png")
        graph_accuracy(np.array(y_eval), np.array(eval_predictions), labels_eval,experiment_path + "General/Eval_accuracy_mass_labels.png")


        plot_loss(solver.losses,solver.eval_losses,experiment_path + '/General/Losses.png')

        # SAVE GENERIC OPTIMIZED FILTER

        first_individual.init_from_trained_filter(solver.model)


        with open(experiment_path + 'General_filter','wb') as Optimized_Filter_Param:
            pickle.dump(first_individual,Optimized_Filter_Param)


    plt.plot(first_individual.f)
    plt.title('Frequency cut_off optimized filter')
    plt.savefig(experiment_path + '/Frequency_cut_off_optimized_filter.png')
    plt.clf()
    plt.close()


    '''
    STEP 4: FEATURE FILTER OPTIMIZATION
    - INITIALIZATION WITH GENERIC OPTIMIZED FILTER
    - Optimization
    - Save graphs
    '''
    print("STEP 4..... FEATURE FILTER OPTIMIZATION")
    centroid_filter = Filter_feature(first_individual,features_train.shape[1],'Filter_spectral_centroid')
    centroid_solver = Solver(centroid_filter,lr = lr)
    centroid_solver.model.to(torch.double)

    train_predictions,eval_predictions = train_solver(centroid_solver,X_train,sample_freq_train,y_train,X_eval,sample_freq_eval,y_eval, n_epochs,features_train,features_eval)

    graph_accuracy(np.array(y_train), np.array(train_predictions), v_train,experiment_path + "Feature/Train_accuracy_v_labels.png")
    graph_accuracy(np.array(y_eval), np.array(eval_predictions), v_eval,experiment_path + "Feature/Eval_accuracy_v_labels.png")

    graph_accuracy(np.array(y_train), np.array(train_predictions), labels_train,experiment_path + "Feature/Train_accuracy_mass_labels.png")
    graph_accuracy(np.array(y_eval), np.array(eval_predictions), labels_eval,experiment_path + "Feature/Eval_accuracy_mass_labels.png")

    plot_loss(centroid_solver.losses,centroid_solver.eval_losses,experiment_path + '/Feature/Losses.png')

    experiment_dict = {'n_epochs':n_epochs,'length':length,'params':params,'train_labels':train_labels,'eval_labels':eval_labels,'lr':lr}
    with open(experiment_path + 'README.txt', 'w') as f:
        print(experiment_dict, file=f)
