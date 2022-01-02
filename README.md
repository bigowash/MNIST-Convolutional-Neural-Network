-unfinished-

# ML-MNIST-Dataset
Using the MNIST dataset to develop a machine learning algorithm that can "read" numbers. 
The aim of this project was to learn make a clean Convolutional Network Machine Learning Algorithm in Java. 
To fully understand how a neural network functions, I did not want to depend on modules that permitted a comfortable level of abstraction like the ones in python using numpy.
This allowed me to manipulate the matricies directly involved in the network as well as the back propogation.

# Implementation

## Set up

The entire algorithm is controlled from the Main function in MNIST_Dataset/Files/Main.java. First the set and network are initialized:
* set : loading of all the images (either TRAINING or TESTING)

`MNISTSet set = MNIST.load(TRAINING);`
* network : contains the hidden layers 

`MNISTNetwork network = new MNISTNetwork(784, 100, 50, 10);`

Most of the parameters can be changed with the constants file.


## Training of Network

