#This class prepares the data that the client class uses for local updates.

import numpy as np
import gzip
# Gzip is used to handle compressed files.
# Analogy: Imagine you have a big textbook, but you want to carry only a small zip file version of it. Gzip helps compress or decompress files to save space.

import os
#OS module helps interact with your computer’s operating system.
#Check what files exist in a folder.   Create new folders.    Run system commands.


import platform
#Platform helps you identify details about the system you're running the code on.

import pickle
#Pickle is a module to save and load Python objects (like dictionaries or lists).


class GetDataSet(object):

  #he GetDataSet class is like a data manager for your machine learning project. 
  # Its job is to load, organize, and prepare data 
  
    def __init__(self, dataSetName, isIID):

        self.name = dataSetName
        self.train_data = None
        self.train_label = None
        self.train_data_size = None
        self.test_data = None
        self.test_label = None
        self.test_data_size = None
        # Training data (used to teach the model).
        # Labels (answers that match the training data).
        # Test data (used to check how well the model learned).
        # Test labels (answers for the test data).

        self._index_in_train_epoch = 0

        #This keeps track of how far we’ve gone through the training data.

        if self.name == 'mnist':  # If the dataset name is 'mnist'
            self.mnistDataSetConstruct(isIID)
        else:
            pass


    def mnistDataSetConstruct(self, isIID):
#ts job is to fetch, check, and prepare MNIST data (a famous dataset of handwritten digits) so the models and clients can use it for training.



        data_dir = r'.\data\MNIST'
        # data_dir = r'./data/MNIST'
        train_images_path = os.path.join(data_dir, 'train-images-idx3-ubyte.gz')
        train_labels_path = os.path.join(data_dir, 'train-labels-idx1-ubyte.gz')
        test_images_path = os.path.join(data_dir, 't10k-images-idx3-ubyte.gz')
        test_labels_path = os.path.join(data_dir, 't10k-labels-idx1-ubyte.gz')
        # setting up path for train and test images and label

        train_images = extract_images(train_images_path)
        train_labels = extract_labels(train_labels_path)
        test_images = extract_images(test_images_path)
        test_labels = extract_labels(test_labels_path)

        # extracting / reading data from the file

        assert train_images.shape[0] == train_labels.shape[0]
        assert test_images.shape[0] == test_labels.shape[0]
        #Ensures that the number of images matches the number of labels.
        #If not, it stops the program.

        self.train_data_size = train_images.shape[0]
        self.test_data_size = test_images.shape[0]
        #Saves the number of training and test images.


        assert train_images.shape[3] == 1
        assert test_images.shape[3] == 1

        #Ensures that the fourth dimension is 1, meaning the images have only one channel(gray scale) other wise will raise assertion error


        train_images = train_images.reshape(train_images.shape[0], train_images.shape[1] * train_images.shape[2])
        test_images = test_images.reshape(test_images.shape[0], test_images.shape[1] * test_images.shape[2])
        #ormalizes pixel values between 0 and 1.
        train_images = train_images.astype(np.float32)
        train_images = np.multiply(train_images, 1.0 / 255.0)
        test_images = test_images.astype(np.float32)
        test_images = np.multiply(test_images, 1.0 / 255.0)


        if isIID:
            order = np.arange(self.train_data_size)
            np.random.shuffle(order)
            self.train_data = train_images[order]
            self.train_label = train_labels[order]
        else:
            labels = np.argmax(train_labels, axis=1)
            order = np.argsort(labels)
            self.train_data = train_images[order]
            self.train_label = train_labels[order]

            #If IID is True, the data is randomized.
           # If not, the data is sorted by labels.



        self.test_data = test_images
        self.test_label = test_labels


def _read32(bytestream):
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]

    #Reads the first 32 bits from a file in a big-endian format 



#This code is a helper script to work with the MNIST dataset
def extract_images(filename):#reads and processes image files 



    """Extract the images into a 4D uint8 numpy array [index, y, x, depth]."""
#  index : Number of images.
#   height, width: Dimensions of each image.
#   depth: 1 (since these are grayscale images).

    print('Extracting', filename)
    with gzip.open(filename) as bytestream:


        magic = _read32(bytestream)  #is a num used to varify the file format 
        if magic != 2051:
            raise ValueError(
                    'Invalid magic number %d in MNIST image file: %s' %
                    (magic, filename))
            

        num_images = _read32(bytestream)
        rows = _read32(bytestream)
        cols = _read32(bytestream)
        buf = bytestream.read(rows * cols * num_images)
        data = np.frombuffer(buf, dtype=np.uint8)
        data = data.reshape(num_images, rows, cols, 1)

        #Reads all pixel data and reshapes it into images.
        return data


def dense_to_one_hot(labels_dense, num_classes=10):
    """Convert class labels from scalars to one-hot vectors."""

    #  Instead of using a single number (e.g., 3) to represent a digit, it creates a vector where:  All positions are 0 except for the one representing the number:
    """ Example: 3 → [0, 0, 0, 1, 0, 0, 0, 0, 0, 0].
       Why do this?

      It's useful for machine learning models because they can handle vectors better than single numbers."""


    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def extract_labels(filename):
    """Extract the labels into a 1D uint8 numpy array [index]."""
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        magic = _read32(bytestream)
        if magic != 2049:
            raise ValueError(
                    'Invalid magic number %d in MNIST label file: %s' %
                    (magic, filename))
        num_items = _read32(bytestream)
        buf = bytestream.read(num_items)
        labels = np.frombuffer(buf, dtype=np.uint8)
        return dense_to_one_hot(labels)


if __name__=="__main__":
    'test data set'
    mnistDataSet = GetDataSet('mnist', True) # It loads the MNIST dataset with IID data (data is shuffled randomly among clients).

    if type(mnistDataSet.train_data) is np.ndarray and type(mnistDataSet.test_data) is np.ndarray and \
            type(mnistDataSet.train_label) is np.ndarray and type(mnistDataSet.test_label) is np.ndarray:
        print('the type of data is numpy ndarray')
    else:
        print('the type of data is not numpy ndarray')
        
    print('the shape of the train data set is {}'.format(mnistDataSet.train_data.shape))
    print('the shape of the test data set is {}'.format(mnistDataSet.test_data.shape))
    print(mnistDataSet.train_label[0:100], mnistDataSet.train_label[11000:11100])
