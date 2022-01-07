package MNIST_Dataset.Files;
import java.util.Arrays;
import java.util.Random;
import java.util.stream.IntStream;

import static MNIST_Dataset.Files.Constants.*;

public class MNISTNetwork {

    public final int[] NETWORK_SIZES;

    public double[][][][] FILTERS;
    public double[][][] NEURONS;
    public double[][] BIAS;

    public double[][] NEURONS_VALUE;

    public double[][] BIAS_CHANGE;
    public double[][][] NEURONS_CHANGE;
    public double[][][][] FILTERS_CHANGE;
    public double[][][][] CONV_BIAS;
    public double[][][][] CONV_BIAS_CHANGE;

    public final int NUM_CONV_LAYERS;
    public final int NUM_DENSE_LAYERS;

    public int[][][] CONV_OUTPUT_DIMENSIONS; // [layer][type of output (0 for conv or 1 for pool)][dimensions]
    public double[][][][][] CONV_OUTPUTS; // [layer][0 for conv, 1 for pool][....rest]

    public double[][] IMAGE;

    public double[][][][][] CHANGE_RECORD_CONV = new double[BATCH_SIZE][][][][];
    public double[][][][][] CHANGE_RECORD_CONV_BIAS = new double[BATCH_SIZE][][][][];
    public double[][][][] CHANGE_RECORD_FC = new double[BATCH_SIZE][][][];
    public double[][][] CHANGE_RECORD_FC_BIAS = new double[BATCH_SIZE][][];

        // Conv filter change
        // conv bias change
        // neuron weight change
        // neuron bias change

    public MNISTNetwork(int... NETWORK_SIZES){

        this.NETWORK_SIZES = NETWORK_SIZES;
        this.NUM_DENSE_LAYERS = NETWORK_SIZES[NETWORK_SIZES.length-1];
        this.NUM_CONV_LAYERS = NETWORK_SIZES.length - NUM_DENSE_LAYERS - 1;
        this.BIAS = new double[NUM_DENSE_LAYERS+1][];
        this.CONV_BIAS = new double[NUM_CONV_LAYERS][][][];
        this.CONV_BIAS_CHANGE = new double[NUM_CONV_LAYERS][][][];

        // initializing weight matricies
        this.FILTERS = new double[NUM_CONV_LAYERS][][][];
        this.NEURONS = new double[NUM_DENSE_LAYERS+1][][];

        // init values of neurons NOT WEIGHTS
        this.NEURONS_VALUE = new double[NUM_DENSE_LAYERS+1][];

        // intiializing changing of weight matricies
        this.FILTERS_CHANGE = new double[NUM_CONV_LAYERS][][][];
        this.NEURONS_CHANGE = new double[NUM_DENSE_LAYERS+1][][];
        this.BIAS_CHANGE = new double[NUM_DENSE_LAYERS+1][];

        this.CONV_OUTPUT_DIMENSIONS = new int[NUM_CONV_LAYERS][2][];

        // conv filters
        for (int layer = 0; layer < NUM_CONV_LAYERS; layer++) {
            this.FILTERS[layer] = new double[NETWORK_SIZES[layer]][][];
            this.FILTERS_CHANGE[layer] = new double[NETWORK_SIZES[layer]][KERNELSIZE][KERNELSIZE];

            if (layer == 0) {
                this.CONV_OUTPUT_DIMENSIONS[layer][0] = new int[]{NETWORK_SIZES[layer], outputConvDimensions(28, KERNELSIZE), outputConvDimensions(28, KERNELSIZE)};
                this.CONV_OUTPUT_DIMENSIONS[layer][1] = new int[]{NETWORK_SIZES[layer], outputPoolDimensions(CONV_OUTPUT_DIMENSIONS[layer][0][1], POOLSIZE), outputPoolDimensions(CONV_OUTPUT_DIMENSIONS[layer][0][1], POOLSIZE)};
            } else {
                this.CONV_OUTPUT_DIMENSIONS[layer][0] = new int[]{NETWORK_SIZES[layer]*NETWORK_SIZES[layer-1], outputConvDimensions(CONV_OUTPUT_DIMENSIONS[layer-1][1][1], KERNELSIZE), outputConvDimensions(CONV_OUTPUT_DIMENSIONS[layer-1][1][1], KERNELSIZE)};
                this.CONV_OUTPUT_DIMENSIONS[layer][1] = new int[]{NETWORK_SIZES[layer]*NETWORK_SIZES[layer-1], outputPoolDimensions(CONV_OUTPUT_DIMENSIONS[layer][0][1], POOLSIZE), outputPoolDimensions(CONV_OUTPUT_DIMENSIONS[layer][0][1], POOLSIZE)};
            }

            this.CONV_BIAS[layer] = new double[CONV_OUTPUT_DIMENSIONS[layer][0][0]][][];
            this.CONV_BIAS[layer] = MNISTNetworkTools.createRandomArray(CONV_OUTPUT_DIMENSIONS[layer][0][0], CONV_OUTPUT_DIMENSIONS[layer][0][2], CONV_OUTPUT_DIMENSIONS[layer][0][1], CONV_BIAS_MIN_VAL, CONV_BIAS_MAX_VAL);
            this.CONV_BIAS_CHANGE[layer] = new double[CONV_BIAS[layer].length][CONV_BIAS[layer][0].length][CONV_BIAS[layer][0][0].length];

            for (int filter = 0; filter < NETWORK_SIZES[layer]; filter++) {
                this.FILTERS[layer][filter] = MNISTNetworkTools.createRandomArray(KERNELSIZE, KERNELSIZE, KERNAL_MIN_VAL, KERNAL_MAX_VAL);
//                this.CONV_BIAS[layer][filter] = MNISTNetworkTools.createRandomArray(CONV_OUTPUT_DIMENSIONS[layer][0][2], CONV_OUTPUT_DIMENSIONS[layer][0][1], CONV_BIAS_MIN_VAL, CONV_BIAS_MAX_VAL);
            }
        }

        // dense layers
        for (int layer = 0; layer < NUM_DENSE_LAYERS+1; layer++) {
            if (layer == 0){
                // get the number of layers from the flattened layers.
                int outputSize = 28;
                for (int i = 0; i < NUM_CONV_LAYERS; i++) {
                    outputSize = (outputSize + 2*PADDING - KERNELSIZE) / CONVSTRIDE + 1;
                    outputSize = (outputSize - POOLSIZE) / POOLSTRIDE + 1;
                }
                outputSize *= outputSize;

                for (int i = 0; i < NUM_CONV_LAYERS; i++) {
                    outputSize *= NETWORK_SIZES[i];
                }

                this.NEURONS_VALUE[layer] = new double[outputSize];

                this.CONV_OUTPUTS = new double[NUM_CONV_LAYERS][2][][][];

                this.NEURONS[layer] = MNISTNetworkTools.createRandomArray(NETWORK_SIZES[NUM_CONV_LAYERS+layer], outputSize, NEURON_MIN_VAL, NEURON_MAX_VAL);
                this.BIAS[layer] = MNISTNetworkTools.createRandomArray(NETWORK_SIZES[NUM_CONV_LAYERS+layer], NEURON_BIAS_MIN_VAL, NEURON_BIAS_MAX_VAL);

                this.NEURONS_CHANGE[layer] = new double[NETWORK_SIZES[NUM_CONV_LAYERS+layer]][outputSize];
                this.BIAS_CHANGE[layer] = new double[NETWORK_SIZES[NUM_CONV_LAYERS+layer]];
            } else if (layer == NUM_DENSE_LAYERS) {
                // matrix needs to map to the desired output (vector size 10)
                this.NEURONS[layer] = MNISTNetworkTools.createRandomArray(10, NETWORK_SIZES[NUM_CONV_LAYERS+layer-1], NEURON_MIN_VAL, NEURON_MAX_VAL);
                this.BIAS[layer] = MNISTNetworkTools.createRandomArray(10, NEURON_MIN_VAL, NEURON_MAX_VAL);

                this.NEURONS_CHANGE[layer] = new double[10][NETWORK_SIZES[NUM_CONV_LAYERS+layer-1]];
                this.BIAS_CHANGE[layer] = new double[10];
            } else {

                this.NEURONS_VALUE[layer] = new double[NETWORK_SIZES[NUM_CONV_LAYERS+layer]];

                this.BIAS[layer] = MNISTNetworkTools.createRandomArray(NETWORK_SIZES[NUM_CONV_LAYERS+layer], NEURON_MIN_VAL, NEURON_MAX_VAL);
                this.NEURONS[layer] = MNISTNetworkTools.createRandomArray(NETWORK_SIZES[NUM_CONV_LAYERS+layer],NETWORK_SIZES[NUM_CONV_LAYERS+layer-1], NEURON_MIN_VAL, NEURON_MAX_VAL);

                this.NEURONS_CHANGE[layer] = new double[NETWORK_SIZES[NUM_CONV_LAYERS+layer]][NETWORK_SIZES[NUM_CONV_LAYERS+layer-1]];
                this.BIAS_CHANGE[layer] = new double[NETWORK_SIZES[NUM_CONV_LAYERS+layer]];
            }
        }
    }

    private int outputConvDimensions(int input_volume, int kernel_size){
        return (input_volume-kernel_size+2*PADDING)/CONVSTRIDE + 1;
    }

    private int outputPoolDimensions(int input_volume, int pool_size) {
        return (input_volume-pool_size)/POOLSTRIDE + 1;
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

                        MNISTImage image = set.getImage(batch[index]);
                        IMAGE = image.imageDouble();

                        double[][][] poolOutput = new double[FILTERS[0].length][][];
                        double[][][] convOutput;

                        // go through convolution stages with the image
                        for (int layer = 0; layer < NUM_CONV_LAYERS; layer++) {
                            if (layer == 0){
                                // convolution part
                                convOutput = convolution(image, FILTERS[layer]);
                            } else {
                                convOutput = convolution(poolOutput, FILTERS[layer]);
                            }

                            // adding bias before pool layer
                            convOutput = addMatricies(convOutput, CONV_BIAS[layer]);

                            // record outputs
                            this.CONV_OUTPUTS[layer][0] = convOutput;

                            // reLU
                            convOutput = reLU(convOutput);

                            // pooling layer
                            poolOutput = pool(convOutput);

                            // record outputs
                            this.CONV_OUTPUTS[layer][1] = poolOutput;

                            int[] dim = getDimensions(convOutput);
                            int[] dim1 = getDimensions(poolOutput);

//                            if (this.CONV_OUTPUT_DIMENSIONS[layer][0] != dim || this.CONV_OUTPUT_DIMENSIONS[layer][1] != dim1){
////                                break;
//                                int a = 0;
//                            }
                        }

                        // go to the fully connected layer
                        double[] neurons = flatten(poolOutput);
//                        this.CONV_OUTPUT_DIMENSIONS = getDimensions(poolOutput);

                        // go to the dense layers
                        for (int layer = 0; layer < NUM_DENSE_LAYERS+1; layer++) {
                            this.NEURONS_VALUE[layer] = neurons;
                            neurons = fullyconnected(neurons, NEURONS[layer], BIAS[layer]);
                        }

                        /*
                            Say the max is worth 12321333 (big number) for digit 0
                            maybe check the loss function by saying ok the goal should have been
                            [12321333, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                            instead of making neurons [1,0,0,0,0,0,0,0]
                        */

                        // normalize the values
                        neurons = softMax(neurons);

                        int result = maxValue(neurons);
                        int expectation = image.getLabel();

                        resetChanges();

                        // should change the weight matricies
                        backpropogation(image.getLabel(), neurons);

                        CHANGE_RECORD_CONV[index] = copy(FILTERS_CHANGE);
                        CHANGE_RECORD_CONV_BIAS[index] = copy(CONV_BIAS_CHANGE);
                        CHANGE_RECORD_FC[index] = copy(NEURONS_CHANGE);
                        CHANGE_RECORD_FC_BIAS[index] = copy(BIAS_CHANGE);

                    }

                    applyChanges();

                }
            }
        }
    }

    private void applyChanges(){
        double sumC1 = 0;
        double sumC2 = 0;
        double sumFC1 = 0;
        double sumFC2 = 0;

        // conv filters change
        for (int layer = 0; layer < FILTERS.length; layer++) {
            for (int filter = 0; filter < FILTERS[layer].length; filter++) {
                for (int i = 0; i < FILTERS[layer][filter].length; i++) {
                    for (int j = 0; j < FILTERS[layer][filter][i].length; j++) {
                        for (int image = 0; image < BATCH_SIZE; image++) {
                            sumC1 += CHANGE_RECORD_CONV[image][layer][filter][i][j];
                        }
                        FILTERS[layer][filter][i][j] += (sumC1/BATCH_SIZE)*LEARNINGRATE;
                        sumC1 = 0;
                    }
                }
            }
        }

        // conv bias change
        for (int layer = 0; layer < CONV_BIAS.length; layer++) {
            for (int filter = 0; filter < CONV_BIAS[layer].length; filter++) {
                for (int i = 0; i < CONV_BIAS[layer][filter].length; i++) {
                    for (int j = 0; j < CONV_BIAS[layer][filter][i].length; j++) {
                        for (int image = 0; image < BATCH_SIZE; image++) {
                            sumC2 += CHANGE_RECORD_CONV_BIAS[image][layer][filter][i][j];
                        }
                        CONV_BIAS[layer][filter][i][j] += (sumC2/BATCH_SIZE)*LEARNINGRATE;
                        sumC2 = 0;
                    }
                }
            }
        }

            // fc weights change
        for (int layer = 0; layer < NEURONS.length; layer++) {
            for (int i = 0; i < NEURONS[layer].length; i++) {
                for (int j = 0; j < NEURONS[layer][i].length; j++) {
                    for (int image = 0; image < BATCH_SIZE; image++) {
                        sumFC1 += CHANGE_RECORD_FC[image][layer][i][j];
                    }
                    NEURONS[layer][i][j] += (sumFC1/BATCH_SIZE)*LEARNINGRATE;
                    sumFC1 = 0;
                }
            }
        }

        // fc bias change
        for (int layer = 0; layer < BIAS.length; layer++) {
            for (int i = 0; i < BIAS[layer].length; i++) {
                for (int image = 0; image < BATCH_SIZE; image++) {
                    sumFC2 += CHANGE_RECORD_FC_BIAS[image][layer][i];
                }
                BIAS[layer][i] += (sumFC2/BATCH_SIZE)*LEARNINGRATE;
                sumFC2 = 0;
            }
        }

    }

    private void resetChanges(){
        this.FILTERS_CHANGE = new double[FILTERS.length][][][];
        for (int i = 0; i < FILTERS.length; i++) {
            this.FILTERS_CHANGE[i] = new double[FILTERS[i].length][][];
            for (int j = 0; j < FILTERS[i].length; j++) {
                this.FILTERS_CHANGE[i][j] = new double[FILTERS[i][j].length][];
                for (int k = 0; k < FILTERS[i][j].length; k++) {
                    this.FILTERS_CHANGE[i][j][k] = new double[FILTERS[i][j][k].length];
                }
            }
        }
        this.CONV_BIAS_CHANGE = new double[CONV_BIAS.length][][][];
        for (int i = 0; i < CONV_BIAS.length; i++) {
            this.CONV_BIAS_CHANGE[i] = new double[CONV_BIAS[i].length][][];
            for (int j = 0; j < CONV_BIAS[i].length; j++) {
                this.CONV_BIAS_CHANGE[i][j] = new double[CONV_BIAS[i][j].length][];
                for (int k = 0; k < CONV_BIAS[i][j].length; k++) {
                    this.CONV_BIAS_CHANGE[i][j][k] = new double[CONV_BIAS[i][j][k].length];
                }
            }
        }
        this.NEURONS_CHANGE = new double[NEURONS.length][][];
        for (int i = 0; i < NEURONS.length; i++) {
            this.NEURONS_CHANGE[i] = new double[NEURONS[i].length][];
            for (int j = 0; j < NEURONS[i].length; j++) {
                this.NEURONS_CHANGE[i][j] = new double[NEURONS[i][j].length];
            }
        }
        this.BIAS_CHANGE = new double[BIAS.length][];
        for (int i = 0; i < BIAS.length; i++) {
            this.BIAS_CHANGE[i] = new double[BIAS[i].length];
        }
    }

    public double[][][][] copy(double[][][][] input){
        double[][][][] output = new double[input.length][input[0].length][input[0][0].length][input[0][0][0].length];
        for (int i = 0; i < input.length; i++) {
            output[i] = copy(input[i]);
        }
        return output;
    }

    public double[][][] copy(double[][][] input){
        double[][][] output = new double[input.length][input[0].length][input[0][0].length];
        for (int i = 0; i < input.length; i++) {
            output[i] = copy(input[i]);
        }
        return output;
    }

    public double[][] copy(double[][] input){
        double[][] output = new double[input.length][input[0].length];
        for (int i = 0; i < input.length; i++) {
            output[i] = Arrays.copyOf(input[i], input[i].length);
        }
        return output;
    }

    private void backpropogation(int value, double[] output){

        // https://ai.stackexchange.com/questions/11643/how-should-i-implement-the-backward-pass-through-a-flatten-layer-of-a-cnn

        // iterate backwards through the layers
            // softmax
            // Dense layers
                // Biases
                // Fully connected
                //
            // flattening
            // pooling + convolution layers

        // gradient of softmax function
        double[] gradient = backCrossSoft(value, output);

        // Dense layer gradients
        for (int layer = NUM_DENSE_LAYERS; layer >= 0; layer--) {
            double[] newGradient = new double[NEURONS[layer][0].length];
            // iterating through the gradients
            for (int j = 0; j < NEURONS[layer].length; j++) {
                for (int i= 0; i < NEURONS[layer][j].length; i++) {
                    NEURONS_CHANGE[layer][j][i] = gradient[j]*NEURONS_VALUE[layer][i];
                    newGradient[i] += NEURONS[layer][j][i]*gradient[j];
                }
                BIAS_CHANGE[layer][j] = gradient[j];
            }
            gradient = newGradient;
        }

        // reshape the gradient to match matrix out put at the end of the conv layers.
        double[][][] conv_gradient = reshape(gradient, new double[CONV_OUTPUT_DIMENSIONS[CONV_OUTPUT_DIMENSIONS.length-1][1][0]][CONV_OUTPUT_DIMENSIONS[CONV_OUTPUT_DIMENSIONS.length-1][1][1]][CONV_OUTPUT_DIMENSIONS[CONV_OUTPUT_DIMENSIONS.length-1][1][2]]);

        // back propagation through convolutive layers
        for (int layer = NUM_CONV_LAYERS-1; layer >= 0; layer--) {
            // reversing the (max) pooling
            conv_gradient = backPool(conv_gradient, CONV_OUTPUTS[layer][0], CONV_OUTPUT_DIMENSIONS[layer][0]);

            // reversing reLU
            for (int i = 0; i < conv_gradient.length; i++) {
                for (int j = 0; j < conv_gradient[0].length; j++) {
                    for (int k = 0; k < conv_gradient[0][0].length; k++) {
                        if (CONV_OUTPUTS[layer][0][i][j][k] < 0.0){
                            conv_gradient[i][j][k] = 0;
                        }
                    }
                }
            }

            double[][][] tempGrad = new double[0][][];

            try {
                tempGrad = new double[NETWORK_SIZES[layer-1]][CONV_OUTPUT_DIMENSIONS[layer-1][1][1]][CONV_OUTPUT_DIMENSIONS[layer-1][1][2]];
            } catch (ArrayIndexOutOfBoundsException ignored) {
                ;
            }

            // reversing the convolution
            for (int filter = 0; filter < NETWORK_SIZES[layer]; filter++) {

                // get bias change
                this.CONV_BIAS_CHANGE[layer] = conv_gradient;

//                CONV_OUTPUTS[layer][filter] >>>>>>>> double[][] (inputs)


                // get filters/weights change
                if (layer >= 1) {

//                    double[][][] tempGrad = new double[NETWORK_SIZES[layer-1]][CONV_OUTPUT_DIMENSIONS[layer-1][1][1]][CONV_OUTPUT_DIMENSIONS[layer-1][1][2]];

                    for (int og_output = 0; og_output < NETWORK_SIZES[layer-1]; og_output++) {
//                        conv_gradient[filter*NETWORK_SIZES[layer-1]+og_output];
//                        for (int subRow = 0; subRow < this.FILTERS_CHANGE[layer][filter].length; subRow++) {
//                            for (int subCol = 0; subCol < this.FILTERS_CHANGE[layer][filter][subRow].length; subCol++) {
//                                this.FILTERS_CHANGE[layer][filter][subRow][subCol] = backConvolution()
//                            }
//                        }

                        double[][] matrixA = CONV_OUTPUTS[layer-1][1][og_output];
                        double[][] kernelA = conv_gradient[filter*NETWORK_SIZES[layer-1]+og_output];

                        this.FILTERS_CHANGE[layer][filter] = addMatricies(convolve(matrixA, kernelA), FILTERS_CHANGE[layer][filter]);

                        double[][] temp = fullCrossCorrelation(conv_gradient[filter*NETWORK_SIZES[layer-1]+og_output], FILTERS[layer][filter]);

                        tempGrad[og_output] = addMatricies(temp, tempGrad[og_output]);

                    }


                }

                else {

                    // below has not been checked
                    for (int og_output = 0; og_output < 28; og_output++) {

                        double[][] kernelA = conv_gradient[filter];

                        this.FILTERS_CHANGE[layer][filter] = addMatricies(convolve(IMAGE, kernelA), FILTERS_CHANGE[layer][filter]);
                    }
                }

                // finding gradient with respect to inputs

            }

            conv_gradient = tempGrad;



            // maybe instead of iterating through the conv_gradient, iterate through the FILTERCHANGE
//            this.FILTERS_CHANGE[layer];
//            CONV_OUTPUTS[layer]

        }
    }

//    private double[][][] backConvolution(double[][][] matrix, double[][][] filters){
//        double[][][] output = new double[matrix.length* filters.length][][];
//        for (int filter = 0; filter < filters.length; filter++) {
//            for (int layer = 0; layer < matrix.length; layer++) {
//                output[filter*matrix.length+layer] = convolve(matrix[layer], filters[filter]);
//            }
//        }
//        return output;
//    }

//    private int[] backPoolToConv(int index, )

    private double[][] fullCrossCorrelation(double[][] outputGradient, double[][] kernel){

        double[][] rotatedKernel = new double[kernel.length][kernel[0].length];
        for (int i = 0; i < 3; i++) {
            rotatedKernel[i] = Arrays.copyOf(kernel[i], kernel[i].length);
        }

//        double[][] rotatedKernel = kernel;
        // rotate kernel
        rotateMatrix(rotatedKernel);

        // perform full cross correlation with rotated kernel
        //padding will be the length of kernel-1
        int padding = kernel.length-1;

        // adding padding to the image
        double[][] paddedOutputGradient = new double[outputGradient.length+2*padding][outputGradient[0].length+2*padding];
        for (int row = 0; row < paddedOutputGradient.length; row++) {
            if (row >= padding && row < paddedOutputGradient.length - padding) {
                for (int col = 0; col < paddedOutputGradient[0].length; col++) {
                   if (col >= padding && col < paddedOutputGradient[0].length - padding){
                       paddedOutputGradient[row][col] = outputGradient[row-padding][col-padding];
                   }
                }
            }
        }

        int kHeight = kernel.length;
        int kWidth = kernel[0].length;

        int newRow = subVectorSize(paddedOutputGradient.length,kHeight,1);
        int newCol = subVectorSize(paddedOutputGradient[0].length,kWidth,1);

        double[][] newMatrix = new double[newCol][newRow];

        for (int row = 0; row < newRow ; row++) {
            for (int col = 0; col < newCol; col++) {
                double[][] subMatrix = subMatrixTop(paddedOutputGradient, row, col, kernel.length);

                // need to apply the kernel to the subMatrix
                newMatrix[row][col] = applyKernel(subMatrix, rotatedKernel);
            }
        }


        return newMatrix;
    }

    public static void rotateMatrix(double[][] mat) {
//        double[][] mat = new double[arr.length][arr[0].length];
        // base case
        if (mat == null || mat.length == 0) {
            return;
        }

        // `N × N` matrix
        int N = mat.length;

        // rotate the matrix by 180 degrees
        for (int i = 0; i < N /2; i++)
        {
            for (int j = 0; j < N; j++)
            {
                double temp = mat[i][j];
                mat[i][j] = mat[N - i - 1][N - j - 1];
                mat[N - i - 1][N - j - 1] = temp;
            }
        }

        // handle the case when the matrix has odd dimensions
        if (N % 2 == 1)
        {
            for (int j = 0; j < N/2; j++)
            {
                double temp = mat[N/2][j];
                mat[N/2][j] = mat[N/2][N - j - 1];
                mat[N/2][N - j - 1] = temp;
            }
        }

//        return;
    }

    private double[][] addMatricies(double[][] a, double[][] b){
        double[][] newMatrix = new double[a.length][a[0].length];
        for (int layer = 0; layer < a.length; layer++) {
            for (int i = 0; i < a[0].length; i++) {
                    newMatrix[layer][i] = a[layer][i] + b[layer][i];
            }
        }
        return newMatrix;
    }

    private double[][] backConvolution(double[][] input, double[][] kernel){
        int kHeight = kernel.length;
        int kWidth = kernel[0].length;

        int newRow = subVectorSize(input.length+2*PADDING,kHeight,CONVSTRIDE);
        int newCol = subVectorSize(input.length+2*PADDING,kWidth,CONVSTRIDE);

        double[][] newMatrix = new double[newCol][newRow];
//
        for (int row = 0; row < newRow ; row++) {
            for (int col = 0; col < newCol; col++) {
                double[][] subMatrix = subMatrixTop(input, row*CONVSTRIDE, col*CONVSTRIDE, kernel.length);
//
                // need to apply the kernel to the subMatrix
                newMatrix[row][col] = applyKernel(subMatrix, kernel);
            }
        }
        return newMatrix;
    }

    private double[][][] backPool(double[][][] gradient, double[][][] inputs, int[] outputDimensions){
        double[][][] newGradient = new double[outputDimensions[0]][outputDimensions[1]][outputDimensions[2]];

        // DIFFERENT IF !MAX_POOL

        // every
        for (int filtered = 0; filtered < inputs.length; filtered++) {
            int matrixRows = inputs[filtered].length;
            int matrixCols = inputs[filtered][0].length;
            int newRow = subVectorSize(matrixRows,POOLSIZE,POOLSTRIDE);
            int newCol = subVectorSize(matrixCols,POOLSIZE,POOLSTRIDE);

            for (int row = 0; row < newRow; row++) {
                for (int col = 0; col < newCol; col++) {
                    double[][] subMatrix = new double[POOLSIZE][POOLSIZE];
                    for (int subRow = 0; subRow < POOLSIZE; subRow++) {
                        for (int subCol = 0; subCol < POOLSIZE; subCol++) {
                            subMatrix[subRow][subCol] = inputs[filtered][row*POOLSTRIDE+subRow][col*POOLSTRIDE+subCol];
                        }
                    }
                    double[][] newMatrix;

                    if (MAX_POOL){
                        // need to return a matrix of 0 except max value
                        newMatrix = backMaxPool(subMatrix, gradient[filtered][row][col]);
                        int a = 0;
                    } else {
                        // idk - something weird with average
                        int a =0;
                    }

                    // update the new gradients
                    for (int subRow = 0; subRow < POOLSIZE; subRow++) {
                        for (int subCol = 0; subCol < POOLSIZE; subCol++) {
                            newGradient[filtered][row*POOLSTRIDE+subRow][col*POOLSTRIDE+subCol] = newMatrix[subRow][subCol];
                        }
                    }
                }
            }
        }
        return newGradient;
    }

    private double[][][] addMatricies(double[][][] a, double[][][] b){
        double[][][] newMatrix = new double[a.length][a[0].length][a[0][0].length];
        for (int layer = 0; layer < a.length; layer++) {
            for (int i = 0; i < a[0].length; i++) {
                for (int j = 0; j < a[0][0].length; j++) {
                   newMatrix[layer][i][j] = a[layer][i][j] + b[layer][i][j];
                }
            }
        }
        return newMatrix;
    }

    private double[][] backMaxPool(double[][] matrix, double gradient){
        double maxVal = matrix[0][0];
        int[] coords = {0,0};
        for (int row = 0; row < matrix.length; row++) {
            for (int col = 0; col < matrix[0].length; col++) {
                if (maxVal < matrix[row][col]) {
                    maxVal = matrix[row][col];
                    coords = new int[]{row, col};
                }
            }
        }
        double[][] newMatrix = new double[matrix.length][matrix[0].length];
        newMatrix[coords[0]][coords[1]] = gradient;
        return newMatrix;
    }

    private double[][][] reshape(double[] inputMatrix, double[][][] outputMatrix){
        for (int i = 0; i < outputMatrix.length; i++) {
            for (int j = 0; j < outputMatrix[i].length; j++) {
                for (int k = 0; k < outputMatrix[i][j].length; k++) {
                    outputMatrix[i][j][k] = inputMatrix[i*outputMatrix[i].length+j*outputMatrix[i][j].length+k];
                }
            }
        }
        return outputMatrix;
    }

    private int[] getDimensions(double[][][] matrix){
        return new int[]{matrix.length, matrix[0].length, matrix[0][0].length};
    }

    public double[] backCrossSoft(int value, double[] output){
        double[] derivative = new double[10];

        for (int i = 0; i < 10; i++) {
            if (i == value) {
                derivative[i] = output[i] - 1;
            } else {
                derivative[i] = output[i];
            }
        }
        return derivative;
    }

    public double crossEntropyError(int value, double[] output){
        return -Math.log(output[value]);
    }

    private double costMSE(int value, double[] output){
        double[] expectation = new double[10];
        expectation[value] = 1.0;

        double sum = 0;

        for (int i = 0; i < 10; i++) {
            sum += Math.pow(expectation[i] - output[i],2);
        }

        return sum;
    }

    private double[] softMax(double[] neuronValues){
        double[] newNeurons = new double[neuronValues.length];
//        neuronValues = normalize(neuronValues);
        double total = Arrays.stream(neuronValues).map(Math::exp).sum();
        for (int i = 0; i < neuronValues.length; i++) {
            newNeurons[i] = Math.exp(neuronValues[i])/total;
        }
        return newNeurons;
    }

    private double[] normalize(double[] vector){
        double max = 0;
        for (int i = 0; i < vector.length; i++) {
            if (max < vector[i]){
                max = vector[i];
            }
        }
        for (int i = 0; i < vector.length; i++) {
            vector[i] /= max/2;
        }
        return vector;
    }

    private int maxValue(double[] vector){
        double max = 0;
        int val = 0;
        for (int i = 0; i < vector.length; i++) {
            if (max < vector[i]){
                max = vector[i];
                val = i;
            }
        }
        return val;
    }

    private double[] fullyconnected(double[] vector, double[][] matrix, double[] bias){

        // want output to be length 10
        double[] output = new double[matrix.length];

        // matrix dimensions should be output (i/row) x input (j/col)
        if (matrix[0].length == vector.length){
            boolean good = true;
        }

        for (int neuron = 0; neuron < matrix.length; neuron++) {
            double sum = bias[neuron];
            for (int weight = 0; weight < vector.length; weight++) {
                sum += vector[weight]*matrix[neuron][weight];
            }

            output[neuron] = reLU(sum);
        }
        return output;
    }

    private double reLU(double val){
        if (val < 0) {return 0;}
        return val;
    }

    private double[] sigmoid(double[] vector) {
        for (int i = 0; i < vector.length; i++) {
            vector[i] = 1d / ( 1 + Math.exp(-vector[i]));
        }
        return vector;
    }

    private double[] flatten(double[][][] matrix){
        double[] vector = new double[matrix.length*matrix[0].length*matrix[0][0].length];
        for (int layer = 0; layer < matrix.length; layer++) {
            for (int row = 0; row < matrix[layer].length; row++) {
                for (int col = 0; col < matrix[layer][row].length; col++) {
                    vector[layer*matrix[layer].length+row*matrix[layer][row].length+col] = matrix[layer][row][col];
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
        return output;
    }

    private double[][][] convolution(double[][][] matrix, double[][][] filters){
        double[][][] output = new double[matrix.length* filters.length][][];
        for (int filter = 0; filter < filters.length; filter++) {
            for (int layer = 0; layer < matrix.length; layer++) {
                output[filter*matrix.length+layer] = convolve(matrix[layer], filters[filter]);
            }
        }
        return output;
    }

    private double[][] convolve(double[][] matrix, double[][] filter){
        int kHeight = filter.length;
        int kWidth = filter[0].length;

        int newRow = subVectorSize(matrix.length+2*PADDING,kHeight,CONVSTRIDE);
        int newCol = subVectorSize(matrix.length+2*PADDING,kWidth,CONVSTRIDE);

        double[][] newMatrix = new double[newCol][newRow];

        for (int row = 0; row < newRow ; row++) {
            for (int col = 0; col < newCol; col++) {
                double[][] subMatrix = subMatrixTop(matrix, row*CONVSTRIDE, col*CONVSTRIDE, filter.length);

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

    private double[][] subMatrixTop(double[][] matrix, int x, int y, int kernel_size) {
        double[][] newSubMatrix = new double[kernel_size][kernel_size];
        for (int row = 0; row < kernel_size; row++) {
            for (int col = 0; col < kernel_size; col++) {
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
        double[][][] newMatrix = new double[matrix.length][matrix[0].length][matrix[0][0].length];
        for (int layer = 0; layer < matrix.length; layer++) {
            for (int row = 0; row < matrix[layer].length; row++) {
                for (int col = 0; col < matrix[layer][row].length; col++) {
                    if (matrix[layer][row][col] < 0) {newMatrix[layer][row][col] = 0;}
                    else {
                        newMatrix[layer][row][col] = matrix[layer][row][col];
                    }
                }
            }
        }
        return newMatrix;
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
                        val = maxPoolSub(subMatrix);
                    } else {
                        val = meanPoolSub(subMatrix);
                    }
                    newMatrix[row][col] = val;
                }
            }
            output[filtered] = newMatrix;
            if (filtered == 51){
                int a = 0;
            }
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

