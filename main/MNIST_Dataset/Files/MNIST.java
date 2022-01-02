package MNIST_Dataset.Files;

import GZip.GZip;

import javax.swing.text.html.ListView;
import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

public class MNIST {

    private static String userDir = System.getProperty("user.dir");

    /**
     * Return a formatted MNISTSet Images corresponding to the tag
     * @param tag tag representing which set (Training or Testing) should be loaded
     * @return (MNISTSet)
     */
    public static MNISTSet load(MNISTSet.MNISTTags tag){

        List<MNISTImage> MNISTImages = new ArrayList<>();

        try{

            List<Integer[][]> images = getImagesInt(Files.readAllBytes(getPath(tag.images)));
            int[] labels = getLabels(Files.readAllBytes(getPath(tag.labels)));

            for (int i = 0; i < images.size(); ++i){
                MNISTImages.add(new MNISTImage(labels[i],images.get(i)));
            }

        }catch (Exception ignored){
            System.out.println("error reading files");
        }

        return new MNISTSet(MNISTImages);
    }


    /**
     * Return the path of the file corresponding to the given tag. If the file hasn't been decompressed yet, retrieve the compressed version, decompress it and save the result.
     * @param file (MNISTFiles) The tag of the file
     * @return (Path) The path of the file.
     * @throws Exception Throws if the file are missing or another error happen.
     */
    public static Path getPath(MNISTFiles file) throws Exception {

        Path compressedPath = Paths.get(userDir).resolve("MNIST").resolve("Compressed");
        Path uncompressedPath = Paths.get(userDir).resolve("MNIST").resolve("Uncompressed");
        int indexOfBreak = file.name.indexOf("-idx");
        String uncompFilename = file.name.substring(0, indexOfBreak) + "." + file.name.substring(indexOfBreak + 1, file.name.lastIndexOf('.'));
        String compFilename = file.name.substring(0, file.name.lastIndexOf('.')) + ".gz";
        if (!containsFile("Uncompressed", uncompFilename)) {
           if (!containsFile("Compressed", compFilename)) {
                throw new Exception("File is nowhere to be found!");
           } else {
                Path filepath = Paths.get("").resolve("MNIST").resolve("Compressed").resolve(compFilename);
                if(!new File(uncompressedPath.toString()).exists()){
                    new File(uncompressedPath.toString()).mkdir();
                }
                GZip.decompress(filepath.toString(), uncompressedPath.resolve(uncompFilename).toString());
           }
        }
        return uncompressedPath.resolve(uncompFilename);
    }


    /**
     * Check if the file is present in the given directory.
     * @param directory The folder to be scanned
     * @param filename The name of the file
     * @return True if the file is present
     */
    private static boolean containsFile(String directory, String filename) {
        Path compressedPath = Paths.get(userDir).resolve("MNIST").resolve(directory);


        File f = new File(compressedPath.toString());
        if (f.exists() && f.list() != null){
            // For each pathname in the path names array
            for (String path : Objects.requireNonNull(f.list())) {
                // if file exists in the compressed directory
                if (filename.equals(path)) {
                    return true;
                }
            }
        }
        return false;
    }

    /**
     * From raw file bytes return the value of all labels
     * @param input raw bytes from file
     * @return array containing all labels
     */
    public static int[] getLabels(byte[] input) {
        int[] labels = new int[input.length - 8];
        for (int b = 8; b < input.length; b++) {
            labels[b - 8] = input[b];
        }
        return labels;
    }

    public static List<Double[][]> getImages(byte[] input) {
        List<Double[][]> images = new ArrayList();
        List<Integer[][]> imagesInt = new ArrayList();
        imagesInt = getImagesInt(input);
        imagesInt.forEach(value -> {

            Double[][] temp = new Double[28][28];
            for (int row = 0; row < value.length; row++) {
                for (int col = 0; col < value[row].length; col++) {
                    double valueDouble = (double) value[row][col] / 255;
                    temp[row][col] = valueDouble;
                }
            }
            images.add(temp);
        });
        return images;
    }

    /**
     * Given raw byte data of all images, returns readable images in a 2D matrix
     * @param input (byte[]) raw data file
     * @return (List<Integer[][]>) 2D list of images (each row is another image vector of length 784)
     */
    public static List<Integer[][]> getImagesInt(byte[] input){

        List<Integer[][]> images = new ArrayList();

        int x = 0, y = 0;
        Integer[][] image = new Integer[28][28];
        Integer[] row = new Integer[28];

        for (int b = 16; b < input.length; b++) {
            row[x++] = Byte.toUnsignedInt(input[b]);
            if (x == 28){
                image[y++] = Arrays.copyOf(row,28);
                if (y == 28) {
                    images.add(image);
                    image = new Integer[28][28];
                    y=0;
                }
                x=0;
            }

        }
        return images;
    }

    /**
     * Transforms 784 length vector into a 28x28 matrix
     * @param vector (List<Integer>) 784 length vector
     * @return (Integer[][]) 28x28 matrix
     */
    public static Integer[][] vectorToMatrix(List<Integer> vector) {
        Integer[][] matrix = new Integer[28][28];
        for (int i = 0; i < 28; i++) {
            for (int j = 0; j < 28; j++) {
                matrix[i][j] = vector.get((i * 28) + j);
            }
        }
        return matrix;
    }

}

