package MNIST_Dataset.Files;

import ML.Constants;

import java.util.Iterator;
import java.util.List;

public class MNISTSet implements Iterable<MNISTImage>{


    /**
     * Array containing the images.
     */
    private final MNISTImage[] images;

    /**
     * The size of the set
     */
    private final int size;

    public MNISTSet(List<MNISTImage> images) {
        this.images = images.toArray(new MNISTImage[0]);
        this.size = images.size();
    }

    public int size() {
        return size;
    }

    public MNISTImage getImage (int index){
        return images[index];
    }

    @Override
    public Iterator<MNISTImage> iterator() {
        return new Iterator<MNISTImage>() {
            int index = 0;
            @Override
            public boolean hasNext() {
                return index < size;
            }

            @Override
            public MNISTImage next() {
                return images[index++];
            }
        };
    }

    public enum MNISTTags {
        TRAINING (MNISTFiles.TRAIN_IMG,MNISTFiles.TRAIN_LAB),
        TESTING (MNISTFiles.TEST_IMG, MNISTFiles.TEST_LAB);

        public MNISTFiles images;
        public MNISTFiles labels;

        MNISTTags(MNISTFiles images, MNISTFiles labels) {
            this.images = images;
            this.labels = labels;
        }

        public static final List<MNISTTags> ALL = List.of(values());
    }

//    public double Test(MNISTNetwork network){
//
//    }

}
