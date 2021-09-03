import numpy as np
import matplotlib.pyplot as plt


# Istogramma dell'errore. Labels è il valore da confrontare con la predizione (predictions) mentre labels name è il nome da visualizzare su ogni istogramma.

def graph_accuracy(labels,predictions,labels_names = None,file_name = None,BINS = 5):

    if labels_names is None:
        labels_names = labels #If I don't pass the nominal label, use the numeric one as the title of the graph
    unique_labels = set(labels_names)

    plt.figure(figsize=(20, 12), dpi=80, tight_layout = True)

    for i,label in enumerate(unique_labels):
        index = np.argwhere(labels_names == label)


        plt.subplot(len(unique_labels)//10 + 1,10,i+1)


        errors = (labels[index]-predictions[index])

        plt.hist(errors,bins = BINS,histtype ='step')
        plt.title(str(label) + '\n Mean(%.2f) \n Std Dev(%.2f)'  % (round(np.mean(errors),2),round(np.std(errors),2)))

    if file_name is not None:
        plt.savefig(file_name)
        plt.clf()
        plt.close()
    else:
        plt.show()


if __name__ == '__main__':

    a = np.array([5.213912,6.213879,1.1239,10.21321,2.213])
    b = ['Gianni','Morandi','Maia','Mattoni?','MAH']

    labels = np.random.choice(a,100000)
    labels_name = np.random.choice(b, 100000)
    predictions = labels + np.random.normal(0,0.1,100000)

    graph_accuracy(labels,predictions,labels_name)

