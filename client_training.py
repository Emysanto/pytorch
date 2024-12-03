import numpy as np
import torch

from torch.utils.data import TensorDataset
#TensorDataset is a class from PyTorch's utility module torch.utils.data. It helps organize data into a format that's easy to work with for training models.
#TensorDataset takes data (like images or text) and their corresponding labels (what the data should predict), and groups them together.

from torch.utils.data import DataLoader
#DataLoader is another helpful tool from PyTorch. It allows you to load the data in small chunks (called batches) rather than all at once.

from getData import GetDataSet
#This line imports a function or class called GetDataSet from a file named getData.py


class client(object):
    def __init__(self, trainDataSet, dev):
      #This refers to the device where the training will happen (could be CPU or GPU)

        self.train_ds = trainDataSet
      #This line saves the trainDataSet (the local data) inside the client object so it can be used late

        self.dev = dev
        #This saves the device (like CPU or GPU) inside the client object so the model knows where to perform computations.


        self.train_dl = None

        #Here, train_dl is an attribute that will later hold the DataLoader (the helper that loads batches of data).
        #Initially, it's set to None because we don't need to load any data right now.

        self.local_parameters = None
        # will later store the para after training
        #initially none cause no training have happenned

    def localUpdate(self, localEpoch, localBatchSize, Net, lossFun, opti, global_parameters):

      #localEpoch: The number of times the client will go through its dataset.

      #localBatchSize: The number of data samples that will be processed together in one step.

      #Net: This is the model that is being trained.

     #lossFun: The loss function, which helps measure how well the model is performing .

     #opti: The optimizer, which helps adjust the model to make it perform better .

     #global_parameters: The model's parameters received from the central server . The client updates its model based on these.


        Net.load_state_dict(global_parameters, strict=True)
        #Net (the model) is loaded with the global_parameters (the model from the central server).
        
        #strict=True means the model will expect the exact same structure as the global model.





        self.train_dl = DataLoader(self.train_ds, batch_size=localBatchSize, shuffle=True)


        #train_dl is created using DataLoader. This helps load the dataset in small chunks
        for epoch in range(localEpoch):#num of times going thru
            for data, label in self.train_dl:#This loop goes through each batch of data and labels (what the model should predict).
                data, label = data.to(self.dev), label.to(self.dev)#Moves the data and labels to the correct device 

                preds = Net(data)
                #This is where the model makes a prediction based on the data it sees.

                loss = lossFun(preds, label)
                # prediction is compared to the actual label

                loss.backward()
                #loss.backward() is where the model calculates how to adjust its parameters

                opti.step()
                #adjust based on result

                opti.zero_grad()#Resets the gradients

        return Net.state_dict()#method returns the updated model parameters

    def local_val(self):
        pass
        #This is a placeholder for a method that could be used for local validation.


class ClientsGroup(object):
  #This is a class for managing multiple clients
  #The client class is for individual students, but the ClientsGroup manages the whole classroom.



    def __init__(self, dataSetName, isIID, numOfClients, dev):

        self.data_set_name = dataSetName
        self.is_iid = isIID
        #isIID: Whether the dataset is Independent and Identically Distributed (IID)
        self.num_of_clients = numOfClients
        self.dev = dev
        self.clients_set = {}#clients_set is a dictionary to hold all clients

        self.test_data_loader = None
        
        #test_data_loader: Prepares a test data loader for evaluating how well the class (clients) performs.

        self.dataSetBalanceAllocation()
        #dataSetBalanceAllocation(): Balances and assigns data to each client.



##method
    def dataSetBalanceAllocation(self):
      #fetches the data and  If IID is True, the dataset is equally distributed among clients.
        mnistDataSet = GetDataSet(self.data_set_name, self.is_iid)

        test_data = torch.tensor(mnistDataSet.test_data)
        test_label = torch.argmax(torch.tensor(mnistDataSet.test_label), dim=1)
        self.test_data_loader = DataLoader(TensorDataset( test_data, test_label), batch_size=100, shuffle=False)

        #Converts the data into PyTorch tensors.
        #Labels are adjusted to match expected format.
        #A test loader is created, allowing batches of 100 for easy evaluation.







        train_data = mnistDataSet.train_data
        train_label = mnistDataSet.train_label

        shard_size = mnistDataSet.train_data_size // self.num_of_clients // 2    #Divides data into shards so that each client gets a fair share.

        shards_id = np.random.permutation(mnistDataSet.train_data_size // shard_size)  #Random permutation of shards to mix data fairly



        for i in range(self.num_of_clients):
            shards_id1 = shards_id[i * 2]
            shards_id2 = shards_id[i * 2 + 1]

            #Two shards per client to ensure every student gets a balanced amount of data.

            data_shards1 = train_data[shards_id1 * shard_size: shards_id1 * shard_size + shard_size]
            data_shards2 = train_data[shards_id2 * shard_size: shards_id2 * shard_size + shard_size]
            label_shards1 = train_label[shards_id1 * shard_size: shards_id1 * shard_size + shard_size]
            label_shards2 = train_label[shards_id2 * shard_size: shards_id2 * shard_size + shard_size]
            local_data, local_label = np.vstack((data_shards1, data_shards2)), np.vstack((label_shards1, label_shards2))
            local_label = np.argmax(local_label, axis=1)

            #Combines data from two shards into a local dataset for each client.

            someone = client(TensorDataset(torch.tensor(local_data), torch.tensor(local_label)), self.dev)
            self.clients_set['client{}'.format(i)] = someone

           # Creates a client instance for each student and stores it in the dictionary.
                     # Example: "client10" gets its own data and labels.

if __name__=="__main__":
    MyClients = ClientsGroup('mnist', True, 100, 1)
    print(MyClients.clients_set['client10'].train_ds[0:100])
    print(MyClients.clients_set['client11'].train_ds[400:500])



    # Creates a ClientsGroup with:
    # Dataset: MNIST
    # IID distribution: True
    # 100 clients.
    # Prints client data for checking:
    # Client 10's first 100 samples.
    # Client 11's samples from 400 to 500.
