package MNIST_Dataset.Display;

import MNIST_Dataset.Files.MNISTImage;

import java.util.Arrays;


public class Display {

    private static String[] greyscale = new String[256];
    static {
        Arrays.fill(greyscale,0,25,"  ");
        Arrays.fill(greyscale,25,50,". ");
        Arrays.fill(greyscale,50,75,": ");
        Arrays.fill(greyscale,75,100,"- ");
        Arrays.fill(greyscale,100,125,"= ");
        Arrays.fill(greyscale,125,150,"+ ");
        Arrays.fill(greyscale,150,175,"* ");
        Arrays.fill(greyscale,175,200,"# ");
        Arrays.fill(greyscale,200,225,"% ");
        Arrays.fill(greyscale,225,256,"@ ");
    }

    /**
     * Takes an image object and prints it out to console
     * @param image (MNISTImage)
     */
    public static void print (MNISTImage image) {
        for (int[] line :  image.getRaster()){
            StringBuilder temp = new StringBuilder();
            for (int pixel: line) {
                temp.append(greyscale[pixel]);
            }
            System.out.println(temp.toString());
        }
    }

    public static void printLabel (MNISTImage image){
        System.out.println(image.getLabel());
    }
}