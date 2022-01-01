package MNIST_Dataset.Files;

import java.util.List;

public enum MNISTFiles {
    TRAIN_IMG("train-images-idx3-ubyte.gz"),
    TRAIN_LAB("train-labels-idx1-ubyte.gz"),
    TEST_IMG("t10k-images-idx3-ubyte.gz"),
    TEST_LAB("t10k-labels-idx1-ubyte.gz");

    public String name;

    MNISTFiles(String name) {
        this.name = name;
    }

    public static final List<MNISTFiles> ALL = List.of(values());
}