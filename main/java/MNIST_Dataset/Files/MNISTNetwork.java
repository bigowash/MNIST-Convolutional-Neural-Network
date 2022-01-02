package MNIST_Dataset.Files;
import java.util.Arrays;
import java.util.Collections;
import java.util.Random;
import java.util.stream.IntStream;

import static MNIST_Dataset.Files.Constants.*;

public class MNISTNetwork {

    private double[][] output;
    private double[][][] weights;
    private double[][] bias;

    private double[][] error_signal;
    private double[][] output_derivative;

    public final int[] NETWORK_LAYER_SIZES;
    public final int   INPUT_SIZE;
    public final int   OUTPUT_SIZE;
    public final int   NETWORK_SIZE;

    public MNISTNetwork(int... NETWORK_LAYER_SIZES) {
        this.NETWORK_LAYER_SIZES = NETWORK_LAYER_SIZES;
        this.INPUT_SIZE = NETWORK_LAYER_SIZES[0];
        this.NETWORK_SIZE = NETWORK_LAYER_SIZES.length;
        this.OUTPUT_SIZE = NETWORK_LAYER_SIZES[NETWORK_SIZE - 1];

        this.output = new double[NETWORK_SIZE][];
        this.weights = new double[NETWORK_SIZE][][];
        this.bias = new double[NETWORK_SIZE][];

        this.error_signal = new double[NETWORK_SIZE][];
        this.output_derivative = new double[NETWORK_SIZE][];

        for (int i = 0; i < NETWORK_SIZE; i++) {
            this.output[i] = new double[NETWORK_LAYER_SIZES[i]];
            this.error_signal[i] = new double[NETWORK_LAYER_SIZES[i]];
            this.output_derivative[i] = new double[NETWORK_LAYER_SIZES[i]];

            this.bias[i] = MNISTNetworkTools.createRandomArray(NETWORK_LAYER_SIZES[i], -0.5, 0.7);

            if (i > 0) {
                weights[i] = MNISTNetworkTools.createRandomArray(NETWORK_LAYER_SIZES[i], NETWORK_LAYER_SIZES[i - 1], -1, 1);
            }
        }
    }

    public void Train(MNISTSet set){
        for (int e = 0; e < EPOCHS; e++) {
            // create a batch from set
            for (int i = BATCH_SIZE-1; i < set.size(); i += BATCH_SIZE) {
                // create array of image indexes within batch interval
                int[] batch = IntStream.range(i-BATCH_SIZE+1, i+1).toArray();
                // iterate through each batch, loop number of times
                for (int loop = 0; loop < LOOPS; loop++) {
                    batch = shuffleArray(batch);
                    // individually go through each image in the batch
                    for (int index = 0; index < BATCH_SIZE; index++) {
                        MNISTImage image = set.getImage(index);
                        // train the network with the image
                        double[][][] filters = {
                                {{0.0,0.0},{0.0,0.0},{0.0,0.0}},
                                {{0.0,0.0},{0.0,0.0},{0.0,0.0}},
                                {{0.0,0.0},{0.0,0.0},{0.0,0.0}}
                        };
                        convolve(image, filters);
                    }
                }
            }
        }
    }

    // different sized filters? or no
    private double[][][] convolve(MNISTImage image, double[][][] filters){
        double[][][] output = new double[filters.length][][];
        for (int i = 0; i < filters.length; i++) {
            double[][] filter = filters[i];
            output[i] = image.convolve(filter);
        }
        return output;
    }

    public int[] shuffleArray(int[] array) {
        Random rand = new Random();
        for (int i = 0; i < array.length; i++) {
            int randomIndexToSwap = rand.nextInt(array.length);
            int temp = array[randomIndexToSwap];
            array[randomIndexToSwap] = array[i];
            array[i] = temp;
        }
        return array;
    }

}

