import os

import argparse
#  argparse: Helps manage input options (like a menu).

from tqdm import tqdm
#tqdm: Displays a progress bar for loops.

import numpy as np
import torch
import torch.nn.functional as F

from torch import optim 
from Models import Mnist_2NN, Mnist_CNN

from clients import ClientsGroup, client #Manages client-side operations in federated learning.



parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="FedAvg")

#This section allows users to input options when running the script. For example:
  #--gpu 0: Use GPU 0.
  # --num_of_clients 100: Set 100 clients.
  #--model_name mnist_2nn: Choose a specific model.

parser.add_argument('-g', '--gpu', type=str, default='0', help='gpu id to use(e.g. 0,1,2,3)')
parser.add_argument('-nc', '--num_of_clients', type=int, default=100, help='numer of the clients')
parser.add_argument('-cf', '--cfraction', type=float, default=0.1, help='C fraction, 0 means 1 client, 1 means total clients')
parser.add_argument('-E', '--epoch', type=int, default=5, help='local train epoch')
parser.add_argument('-B', '--batchsize', type=int, default=10, help='local train batch size')
parser.add_argument('-mn', '--model_name', type=str, default='mnist_2nn', help='the model to train')
parser.add_argument('-lr', "--learning_rate", type=float, default=0.01, help="learning rate, \
                    use value from origin paper as default")
parser.add_argument('-vf', "--val_freq", type=int, default=5, help="model validation frequency(of communications)")
parser.add_argument('-sf', '--save_freq', type=int, default=20, help='global model save frequency(of communication)')
parser.add_argument('-ncomm', '--num_comm', type=int, default=1000, help='number of communications')
parser.add_argument('-sp', '--save_path', type=str, default='./checkpoints', help='the saving path of checkpoints')
parser.add_argument('-iid', '--IID', type=int, default=0, help='the way to allocate data to clients')


def test_mkdir(path):  # creating dic
    if not os.path.isdir(path):  # checking if folder exist 
        os.mkdir(path)   # if not making one


if __name__=="__main__":
    args = parser.parse_args()  # Gathers user inputs for training, like the number of clients, epochs, etc.
    args = args.__dict__

    test_mkdir(args['save_path'])

    os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu']   # this is setting up device 
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu") # if gpu available then  use gpu else use cpu

    net = None
    if args['model_name'] == 'mnist_2nn':    # model selection
        net = Mnist_2NN()
    elif args['model_name'] == 'mnist_cnn':
        net = Mnist_CNN()

    if torch.cuda.device_count() > 1:  #If multiple GPUs are available, use them all to speed up training. 
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = torch.nn.DataParallel(net)
    net = net.to(dev)



    loss_func = F.cross_entropy
    opti = optim.SGD(net.parameters(), lr=args['learning_rate'])  #Optimizer: Updates the model during training using Stochastic Gradient Descent (SGD)

    myClients = ClientsGroup('mnist', args['IID'], args['num_of_clients'], dev)  #creates a group of clients that will train the model using MNIST data.
    testDataLoader = myClients.test_data_loader  # data is loded





    num_in_comm = int(max(args['num_of_clients'] * args['cfraction'], 1))  #Determines how many clients participate in each communication round.

    global_parameters = {}
    for key, var in net.state_dict().items():
        global_parameters[key] = var.clone()

    for i in range(args['num_comm']):   #the loop simulates communication rounds between the server and clients.
        print("communicate round {}".format(i+1))


        order = np.random.permutation(args['num_of_clients'])   # random selection
        clients_in_comm = ['client{}'.format(i) for i in order[0:num_in_comm]]

        sum_parameters = None
        for client in tqdm(clients_in_comm):
            local_parameters = myClients.clients_set[client].localUpdate(args['epoch'], args['batchsize'], net,
                                                                         loss_func, opti, global_parameters)    
            # aggregating para

            if sum_parameters is None:
                sum_parameters = {}
                for key, var in local_parameters.items():
                    sum_parameters[key] = var.clone()
            else:
                for var in sum_parameters:
                    sum_parameters[var] = sum_parameters[var] + local_parameters[var]

              # aggegating 

        for var in global_parameters:
            global_parameters[var] = (sum_parameters[var] / num_in_comm)  # averaging 

        with torch.no_grad():
            if (i + 1) % args['val_freq'] == 0:
                net.load_state_dict(global_parameters, strict=True)  # calc  accuracy

                # evaluating model without changing para
                sum_accu = 0
                num = 0
                for data, label in testDataLoader:
                    data, label = data.to(dev), label.to(dev)
                    preds = net(data)
                    preds = torch.argmax(preds, dim=1)
                    sum_accu += (preds == label).float().mean()
                    num += 1
                print('accuracy: {}'.format(sum_accu / num))

        if (i + 1) % args['save_freq'] == 0:   # saving model
            torch.save(net, os.path.join(args['save_path'],
                                         '{}_num_comm{}_E{}_B{}_lr{}_num_clients{}_cf{}'.format(args['model_name'],
                                                                                                i, args['epoch'],
                                                                                                args['batchsize'],
                                                                                                args['learning_rate'],
                                                                                                args['num_of_clients'],
                                                                                                args['cfraction'])))
