package MNIST_Dataset.Files;

public class Constants {

    final static double ALPHA = 0.01;
    final static double BETA = 0.95;

    final static int BATCH_SIZE = 100;
    final static int LOOPS = 50;
    final static int EPOCHS = 20;

    final static double LEARNINGRATE = 0.3;
    final static int PADDING = 1;
    final static int CONVSTRIDE = 2;
    final static double PADDINGVALUE = 0.0;

    final static int KERNELSIZE = 3; // same value for all kernals regardless of conv layer
    final static double KERNAL_MIN_VAL = -0.5;
    final static double KERNAL_MAX_VAL = 0.7;

    final static double NEURON_MIN_VAL = -0.5;
    final static double NEURON_MAX_VAL = 0.7;

    final static boolean MAX_POOL = true; // Average Pool (false)
    final static int POOLSIZE = 3;
    final static int POOLSTRIDE = 2;
}
