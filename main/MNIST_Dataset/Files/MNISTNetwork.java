package MNIST_Dataset.Files;
import java.util.Arrays;
import java.util.Collections;
import java.util.Random;
import java.util.stream.IntStream;

import static MNIST_Dataset.Files.Constants.*;

public class MNISTNetwork {

    public final int[] NETWORK_SIZES;

    private double[][][][] FILTERS;
    private double[] NEURONS;

    public final int NUM_LAYERS;
    public final int NUM_NEURONS;

    public MNISTNetwork(int... NETWORK_SIZES){

        this.NETWORK_SIZES = NETWORK_SIZES;
        this.NUM_LAYERS = NETWORK_SIZES.length - 1;
        this.NUM_NEURONS = NETWORK_SIZES[NUM_LAYERS];

        this.FILTERS = new double[NUM_LAYERS][][][];

        for (int layer = 0; layer < NUM_LAYERS; layer++) {
            this.FILTERS[layer] = new double[NETWORK_SIZES[layer]][][];
            for (int filter = 0; filter < NETWORK_SIZES[layer]; filter++) {
                this.FILTERS[layer][filter] = new double[KERNELSIZE][];
                for (int kernal_row = 0; kernal_row < KERNELSIZE; kernal_row++) {
                    this.FILTERS[layer][filter][kernal_row] = MNISTNetworkTools.createRandomArray(KERNELSIZE, KERNAL_MIN_VAL, KERNAL_MAX_VAL);
                }
            }
        }

        this.NEURONS = new double[NUM_NEURONS];
        this.NEURONS = MNISTNetworkTools.createRandomArray(NUM_NEURONS, NEURON_MIN_VAL, NEURON_MAX_VAL);
    }

//    public MNISTNetwork(int... NETWORK_LAYER_SIZES) {
//        this.NETWORK_LAYER_SIZES = NETWORK_LAYER_SIZES;
//        this.INPUT_SIZE = NETWORK_LAYER_SIZES[0];
//        this.NETWORK_SIZE = NETWORK_LAYER_SIZES.length;
//        this.OUTPUT_SIZE = NETWORK_LAYER_SIZES[NETWORK_SIZE - 1];
//
//        this.output = new double[NETWORK_SIZE][];
//        this.weights = new double[NETWORK_SIZE][][];
//        this.bias = new double[NETWORK_SIZE][];
//
//        this.error_signal = new double[NETWORK_SIZE][];
//        this.output_derivative = new double[NETWORK_SIZE][];
//
//        for (int i = 0; i < NETWORK_SIZE; i++) {
//            this.output[i] = new double[NETWORK_LAYER_SIZES[i]];
//            this.error_signal[i] = new double[NETWORK_LAYER_SIZES[i]];
//            this.output_derivative[i] = new double[NETWORK_LAYER_SIZES[i]];
//
//            this.bias[i] = MNISTNetworkTools.createRandomArray(NETWORK_LAYER_SIZES[i], -0.5, 0.7);
//
//            if (i > 0) {
//                weights[i] = MNISTNetworkTools.createRandomArray(NETWORK_LAYER_SIZES[i], NETWORK_LAYER_SIZES[i - 1], -1, 1);
//            }
//        }
//    }

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

                        double[][][] poolOutput = new double[FILTERS[0].length][][];

                        // go through convolution stages with the image
                        for (int layer = 0; layer < NUM_LAYERS; layer++) {
                            double[][][] convOutput;
                            if (layer == 0){
                                // convolution part
                                convOutput = convolution(image, FILTERS[layer]);
                            } else {
                                convOutput = convolution(poolOutput, FILTERS[layer]);
                            }
                            // pooling part
                            poolOutput = pool(convOutput);
                        }

                        // go to the fully connected layer
                        // map double[][][] to double[]
                        double[] neurons = flatten(poolOutput);
                    }
                }
            }
        }
    }

    private double[] flatten(double[][][] matrix){
        double[] vector = new double[matrix.length*matrix[0].length*matrix[0][0].length];
        for (int layer = 0; layer < matrix.length; layer++) {
            for (int row = 0; row < matrix[0].length; row++) {
                for (int col = 0; col < matrix[0][0].length; col++) {
                    vector[layer*row*col] = matrix[layer][row][col];
                }
            }
        }
    return vector;
    }

    private double[][][] convolution(MNISTImage image, double[][][] filters){
        double[][][] output = new double[filters.length][][];
        for (int i = 0; i < filters.length; i++) {
            double[][] filter = filters[i];
//            output[i] = image.convolve(filter);
             output[i] = convolve(image.imageDouble(), filter);
        }
        return reLU(output);
    }

    private double[][][] convolution(double[][][] matrix, double[][][] filters){
        double[][][] output = new double[matrix.length* filters.length][][];
        for (int filter = 0; filter < filters.length; filter++) {
            for (int layer = 0; layer < matrix.length; layer++) {
                output[filter*layer+layer] = convolve(matrix[layer], filters[filter]);
            }
        }
        return output;
    }

    private double[][] convolve(double[][] matrix, double[][] filter){
        int kHeight = filter.length;
        int kWidth = filter[0].length;

        int newRow = subVectorSize(28+2*PADDING,kHeight,CONVSTRIDE);
        int newCol = subVectorSize(28+2*PADDING,kWidth,CONVSTRIDE);

        double[][] newMatrix = new double[newCol][newRow];

        for (int row = 0; row < newRow ; row++) {
            for (int col = 0; col < newCol; col++) {
                double[][] subMatrix = subMatrixTop(matrix, row*CONVSTRIDE, col*CONVSTRIDE);

                // need to apply the kernel to the subMatrix
                newMatrix[row][col] = applyKernel(subMatrix, filter);
            }
        }
        return newMatrix;
    }

    private double applyKernel(double[][] matrix, double[][] kernel){
        double sum = 0;
        for (int row = 0; row < matrix.length; row++) {
            for (int col = 0; col < matrix[0].length; col++) {
                sum += matrix[row][col]*kernel[row][col];
            }
        }
        return sum;
    }

    private double[][] subMatrixTop(double[][] matrix, int x, int y) {
        double[][] newSubMatrix = new double[KERNELSIZE][KERNELSIZE];
        for (int row = 0; row < KERNELSIZE; row++) {
            for (int col = 0; col < KERNELSIZE; col++) {
                try {
                    newSubMatrix[row][col] = matrix[x+row][y+col];
                } catch(IndexOutOfBoundsException e) {
                    newSubMatrix[row][col] = PADDINGVALUE;
                }
            }
        }
        return newSubMatrix;
    }

    private double[][][] reLU(double[][][] matrix){
        for (int layer = 0; layer < matrix.length; layer++) {
            for (int row = 0; row < matrix[layer].length; row++) {
                for (int col = 0; col < matrix[layer][row].length; col++) {
                    if (matrix[layer][row][col] < 0) matrix[layer][row][col] = 0;
                }
            }
        }
        return matrix;
    }

    private double[][][] pool(double[][][] matrix){
        double[][][] output = new double[matrix.length][][];

        for (int filtered = 0; filtered < matrix.length; filtered++) {
            int matrixRows = matrix[filtered].length;
            int matrixCols = matrix[filtered][0].length;
            int newRow = subVectorSize(matrixRows,POOLSIZE,POOLSTRIDE);
            int newCol = subVectorSize(matrixCols,POOLSIZE,POOLSTRIDE);

            double[][] newMatrix = new double[newCol][newRow];

            double val = 0;

            for (int row = 0; row < newRow; row++) {
                for (int col = 0; col < newCol; col++) {
                    double[][] subMatrix = new double[POOLSIZE][POOLSIZE];
                    for (int subRow = 0; subRow < POOLSIZE; subRow++) {
                        for (int subCol = 0; subCol < POOLSIZE; subCol++) {
                            subMatrix[subRow][subCol] = matrix[filtered][row*POOLSTRIDE+subRow][col*POOLSTRIDE+subCol];
                        }
                    }
                    if (MAX_POOL){
                        val = meanPoolSub(subMatrix);
                    } else {
                        val = maxPoolSub(subMatrix);
                    }
                    newMatrix[row][col] = val;
                }
            }
            output[filtered] = newMatrix;
        }
        return output;
    }

    public static double meanPoolSub(double[][] matrix){
        int rows = matrix.length;
        int cols =  matrix[0].length;
        int sum = 0;

        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                sum += matrix[row][col];
            }
        }
        return Math.round((double) sum/ (rows*cols));
    }

    public static double maxPoolSub(double[][] matrix){
        int rows = matrix.length;
        int cols =  matrix[0].length;
        double max = matrix[0][0];

        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                if (matrix[row][col] > max) {
                    max = matrix[row][col];
                }
            }
        }
        return max;
    }

    public static int subVectorSize(int length, int size, int stride){
        int count = 0;
        for (int i = size; i <= length; i+= stride) {
            count++;
        }
        return count;
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

