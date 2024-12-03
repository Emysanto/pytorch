import torch
#performing mathematical operations, and interacting with the GPU for faster computation.

import torch.nn as nn
#to access predefined neural network layers, loss functions, and utilities

import torch.nn.functional as F
#It includes a variety of functions for activation functions, loss functions, and other common neural network operations 
class Mnist_2NN(nn.Module):
  #the class Mnist_2NN is a blueprint for creating a neural network that can learn to recognize handwritten digits 
  #nn.modele will have nueral training ablities

    def __init__(self):
        super().__init__()#subset of nn.Module

        self.fc1 = nn.Linear(784, 200)
        #reducing 784 datas  to 200

        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 10)

    def forward(self, inputs):
        #input is the data(pictures)

        tensor = F.relu(self.fc1(inputs))
        # First, the network processes the image through the first layer (fc1) and applies ReLU, which is like a filter that helps the network "focus" on important features. 
        #ReLU is a function that makes negative values 0 and keeps positive ones unchanged.

        tensor = F.relu(self.fc2(tensor))
        #The result is then processed through the second layer (fc2) with another ReLU activation.

        tensor = self.fc3(tensor)
        #Finally, the result goes through the last layer (fc3). The output is a set of 10 values representing the scores for each digit (0-9). 
        #These scores help decide which digit is the best match for the input image.
        return tensor


class Mnist_CNN(nn.Module):#Convolutional Neural Network (CNN)

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2)
        #conv1: This is a convolutional layer.. It looks at small parts of the image (called receptive fields) and tries to figure out important patterns.

        # in_channels=1: The input image has 1 color channel (since it’s grayscale).
        # out_channels=32: This layer will output 32 feature maps (think of it as detecting 32 different features from the image).
        # kernel_size=5: The filter looks at a 5x5 block of pixels at a time.
        # stride=1: The filter moves one pixel at a time.
        # padding=2: Padding ensures that the filter fits perfectly around the image without cutting off edges.




        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # This is a pooling layer that reduces the size of the feature map. It’s like zooming out on an image,  we use max pooling 

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        #The second convolutional and pooling layers repeat the same steps as the first ones, but this time, the network detects even more complex features


        self.fc1 = nn.Linear(7*7*64, 512)
        #fully connected layer after all the convolutional and pooling steps. The feature maps are flattened (like turning a 2D image into a 1D list), 
        #and the data is passed through this layer, which tries to "learn" from the features detected in the previous layers.

        self.fc2 = nn.Linear(512, 10)
        #The final fully connected layer gives us 10 output values, one for each possible digit (0 to 9). The network will pick the digit with the highest score.

    def forward(self, inputs):
        tensor = inputs.view(-1, 1, 28, 28)# Reshaping the input image
        tensor = F.relu(self.conv1(tensor))

        tensor = self.pool1(tensor)
        
        tensor = F.relu(self.conv2(tensor))
        tensor = self.pool2(tensor)
        tensor = tensor.view(-1, 7*7*64)
        tensor = F.relu(self.fc1(tensor))
        tensor = self.fc2(tensor)
        return tensor

