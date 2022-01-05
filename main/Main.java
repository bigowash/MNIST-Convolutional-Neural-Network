import MNIST_Dataset.Display.Display;
import MNIST_Dataset.Files.*;

import static MNIST_Dataset.Files.MNISTSet.MNISTTags.TESTING;
import static MNIST_Dataset.Files.MNISTSet.MNISTTags.TRAINING;

public class Main {
    // main program function
    public static void main(String[] args) {
        MNISTSet set = MNIST.load(TRAINING);
//        MNISTSet set = MNIST.load(TESTING);

//        MNISTNetwork network = new MNISTNetwork(10, 10, 300);
        MNISTNetwork network = new MNISTNetwork(10, 9, 11, 1);
        MNISTImage image = set.getImage(1);
//        System.out.println(image.getLabel());
//        System.out.println(image.image());

        //train
        network.Train(set);
        //test
//        double accuracy = set.Test(network);

    }
}
