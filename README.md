-unfinished-

# ML-MNIST-Dataset
Using the MNIST dataset to develop a machine learning algorithm that can "read" numbers. 
The aim of this project was to learn make a clean Convolutional Network Machine Learning Algorithm in Java. 
To fully understand how a neural network functions, I did not want to depend on modules that permitted a comfortable level of abstraction like the ones in python using numpy.
This allowed me to manipulate the matricies directly involved in the network as well as the back propogation.

Additionally, a very important feature of the code was its adaptability with respect to the inputs:
* Kernel size
* Stride
* Pooling method
* Number of layers
* Number of filters
* Number of fully connected nodes
* etc
This was, once again, to be able to see what changes affected the efficiency of the code and to what extent.
The eventual goal was to devise a method to automatically optimize the value of these parameters to optimize the analysis.

# Implementation

## Set up

The entire algorithm is controlled from the Main function in MNIST_Dataset/Files/Main.java. First the set and network are initialized:
* set : loading of all the images (either TRAINING or TESTING)

`MNISTSet set = MNIST.load(TRAINING);`
* network : contains the hidden layers 

`MNISTNetwork network = new MNISTNetwork(784, 100, 50, 10);`

Most of the parameters can be changed with the constants file.


## Training : Convolution

## Training : Pooling

## Training : Fully Connected Layer

## Training : Backward Propagation

## Testing of Network

## Visualisation of Network

As the aim of this project was to understand the steps needed to be taken for a neural network. 
And the benefits of using a convoluted neural network is from the spacial/feature recognition. 
Seeing the result of the various convolutive layers is interesting, and to see how they evolved.
[]visualizing CNN