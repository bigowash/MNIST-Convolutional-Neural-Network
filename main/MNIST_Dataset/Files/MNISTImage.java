package MNIST_Dataset.Files;

import java.util.Arrays;

import static MNIST_Dataset.Files.Constants.*;

public class MNISTImage {
    /**
     * The value of the number represented on the image.
     */
    private final int label;

    /**
     * The image represented by a matrix of int values between 0 and 255.
     * The format is image[x][y]
     */
    private final int[][] image;

    public int getLabel() {
        return label;
    }

    public int[][] getRaster() {
        int [][] copy = new int[image.length][];
        for(int i = 0; i < image.length; i++)
            copy[i] = image[i].clone();
        return copy;
    }

//    public MNISTImage(int label, Double[][] image) {
//        double[][] primitiveImg = new double[image[0].length][image.length];
//        for (int i = 0; i < image.length; i++) {
//            for (int j = 0; j < image[0].length; ++j){
//                primitiveImg[i][j] = image[i][j];
//            }
//        }
//        this.label = label;
//        this.image = primitiveImg;
//    }

    public MNISTImage(int label, Integer[][] image) {
        int[][] primitiveImg = new int[image[0].length][image.length];
        for (int i = 0; i < image.length; i++) {
            for (int j = 0; j < image[0].length; ++j){
                primitiveImg[i][j] = image[i][j];
            }
        }
        this.label = label;
        this.image = primitiveImg;
    }

    public int label() {
        return label;
    }

    public int[][] image() {
        return image;
    }

    public double[][] imageDouble(){
        double[][] matrix = new double[28][28];
        for (int row = 0; row < 28; row++) {
            for (int col = 0; col < 28; col++) {
                matrix[row][col] = image[row][col];
            }
        }
        return matrix;
    }

    public enum Color {
        WHITE(0),
        GREY(127),
        BLACK(255),
        EMPTY(-1);

        public int value;

        Color(int value) {
            this.value = value;
        }
    }

    public double[][] convolve(double[][] filter){
        // convolve an image, return
        int kHeight = filter.length;
        int kWidth = filter[0].length;

//        System.out.println(STRIDE);
        int newRow = subVectorSize(28+2*PADDING,kHeight,CONVSTRIDE);
        int newCol = subVectorSize(28+2*PADDING,kWidth,CONVSTRIDE);

        double[][] newMatrix = new double[newCol][newRow];

        for (int row = 0; row < newRow ; row++) {
            for (int col = 0; col < newCol; col++) {
                double[][] subMatrix = subMatrixTop(row*CONVSTRIDE, col*CONVSTRIDE, kWidth, kHeight);
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

    private double[] matrixToVector(double[][] matrix) {
        int rows = matrix.length;
        int cols = matrix[0].length;
        double[] vector = new double[rows * cols];

        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                vector[cols * row + rows * col] = matrix[row][col];
            }
        }
        return vector;
    }

    private double[][] subMatrixTop(int x, int y, int width, int height) {
        double[][] newSubMatrix = new double[height][width];
        for (int row = 0; row < height; row++) {
            for (int col = 0; col < width; col++) {
                try {
                    newSubMatrix[row][col] = (double) image[x+row][y+col];
                } catch(IndexOutOfBoundsException e) {
                    newSubMatrix[row][col] = PADDINGVALUE;
                }
            }
        }
        return newSubMatrix;
    }

    public static int subVectorSize(int length, int size, int stride){
        int count = 0;
        for (int i = size; i <= length; i+= stride) {
            count++;
        }
        return count;
    }
}

