package MNIST_Dataset.Files;

import java.util.Arrays;

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

}

