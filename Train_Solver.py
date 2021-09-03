import numpy as np
import torch

from Solver_Class import Solver
from Individual_Class import Individual

from Filter_param_Class import Filter_param
from Filter_polynomial_Class import Filter_poly
from Visualize_Accuracy import graph_accuracy


import matplotlib.pyplot as plt

#Back props to optimize filter. Loss type can be
# 'last_values': the last M values are compared to the label, the loss is the mean of the M losses
# 'average' : prediction is a weigthed average (with weigths that are learned). The loss is the difference between the average and the label

def plot_loss(train_losses,eval_losses,file_name = None): #  graphs train and eval loss and visualize error divided by label
    plt.yscale('log')
    plt.ylim(np.min(eval_losses),10)
    plt.plot(train_losses,label = 'train')
    plt.plot(eval_losses, label = 'eval')


    plt.title('Train (%.2f), Eval mean error:  (%.2f)' % (round(train_losses[-1], 2), round(eval_losses[-1], 2)))
    plt.legend()

    if file_name is None:
        plt.show()
    else:
        plt.savefig(file_name,dpi = 160)
        plt.clf()
        plt.close()


def train_solver(solver, data, F, labels, eval_data, F_eval, eval_labels, epochs=100,train_features = None,eval_features = None, loss_type = 'last_values'):
    minimum_loss = np.inf

    for i in range(epochs):
        solver.model.train()

        if train_features is None:
            solver.model.forward(data,F)
        else:
            solver.model.forward(data,train_features, F)

        solver.optimizer.zero_grad()

        loss = 0

        if loss_type == 'average':
            solver.prediction = solver.model.prediction
            loss = solver.criterion(solver.model.prediction,labels)
        else:
            M = 5
            solver.prediction = solver.model.y_2[-1]
            for N in range(M):
                loss += solver.criterion(solver.model.y_2[-N - 1], labels) / M

        solver.losses.append(loss.item())
        train_prediction = solver.model.y_2[-1].detach()

        loss.backward()
        solver.optimizer.step()

        solver.model.eval()
        if eval_features is None:
            solver.model.forward(eval_data, F_eval)
        else:
            solver.model.forward(eval_data, eval_features, F_eval)

        eval_loss = 0

        if loss_type == 'average':
            solver.eval_prediction = solver.model.prediction
            eval_loss = solver.criterion(solver.model.prediction,eval_labels)
        else:
            solver.eval_prediction = solver.model.y_2[-1]
            for N in range(M):
                eval_loss += solver.criterion(solver.model.y_2[-N - 1], eval_labels) / M
        solver.eval_losses.append(eval_loss.item())
        eval_prediction = solver.model.y_2[-1].detach()



        aggregated_loss = max(loss.item(),eval_loss.item())
        if aggregated_loss < minimum_loss:
            torch.save(solver.model,'./' + solver.model.name + '_' + loss_type)
            minimum_loss = aggregated_loss


        if i % (2500) == 0 and i!=0:
            plot_loss(solver.losses,solver.eval_losses)

    return np.array(train_prediction),np.array(eval_prediction)

if __name__ == '__main__':
    params = {'f_inf': [0.01, 10], 'f_start': [10, 120], 'f_decay': [0.001, 0.5], 'f_T': [0.0001, 1],
              'w_T': [0.001, 1], 'w_offset': [0, 1.]}

    x = np.random.normal(0,1,size = (len(params.keys())))
    Individual = Individual(params,x,L = 100)
    #filter = Filter_poly(Individual,F = 1000)
    filter = Filter_param(Individual)

    #filter.forward(torch.rand(10,1000)/1000 + 5 + torch.sin(torch.arange(1000)*5.).reshape(-1,1000))

    possible_labels = [1,2,3,4]
    solver = Solver(filter)

    data = torch.rand(100,1000)
    eval_data = torch.rand(100,1000)


    F_sampling_train = 100* torch.ones(100,1)
    F_sampling_eval = 100* torch.ones(100,1)

    labels = torch.FloatTensor(np.random.choice(possible_labels,100))
    eval_labels = torch.FloatTensor(np.random.choice(possible_labels,100))

    train_solver(solver,data,F_sampling_train,labels,eval_data,F_sampling_eval,eval_labels,5)

    if filter.name == 'Filter_param':
        plt.plot(solver.model.f.detach())
    else:
        for i in range(len(solver.model.f)):
            plt.plot(solver.model.f[i].detach())
    plt.show()