package global.melanomaDetector;

import global.melanomaDetector.GetPropValuesHelper;
import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.io.labels.PathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.recordreader.objdetect.ObjectDetectionRecordReader;
import org.datavec.image.recordreader.objdetect.impl.VocLabelProvider;
import org.datavec.image.transform.ImageTransform;
import org.datavec.image.transform.PipelineImageTransform;
import org.datavec.image.transform.ResizeImageTransform;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.CompositeDataSetPreProcessor;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.dataset.api.preprocessor.classimbalance.UnderSamplingByMaskingPreProcessor;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.shade.protobuf.compiler.PluginProtos;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class SkinDatasetIterator {
    private static final Logger log = LoggerFactory.getLogger(SkinDatasetIterator.class);

    //dataDir is path to folder containing dl4j datasets on local machine
    //parentDir is path to folder containing train and test data sets
    //trainDir and testDir refer to the specific folder containing the train & test data resp.
    private static String dataDir;
    public static String parentDir;
    private static Path trainDir, testDir;
    //Define name of folders containing train and test data
    public static String trainfolder ="train";
    public static String testfolder ="test";
    //public static String testfolder ="validation";


    //Random number to initialise FileSplit when it works on trainData and testData
    private static final int seed = 123;
    private static Random rng = new Random(seed);
    private static final int channels = 3;
    private static final int numClasses = 2;
    private static int batchSize;
    private static final int height = 224; //VGG takes in inputs of 32*7 X 32*7
    private static final int width = 224;
    private static final String [] allowedExtensions = NativeImageLoader.ALLOWED_FORMATS;

    //scale input to 0 - 1
    private static DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
    //Labels incoming images by name of containing folder.
    //Ex: parentDir = data/train & subfolders containing images in parentDir are "melanoma" and "not_melanoma"
    //Thus, all images are labelled as either "melanoma" and "not_melanoma"
    private static ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
    private static ImageTransform transform;
    private static InputSplit trainData,testData;


    //setup() and loadData() prepare data to be loaded into the model
    //setup() is called in MelanomaDetector.java, before trainIterator() and testIterator()
    //loadData() obtains path to source image folders
    private static void loadData() throws IOException{
        //dataDir creates a path of "C:\Users\win10AccountName\.deeplearning4j\data"
        dataDir= Paths.get(
                System.getProperty("user.home"),
                GetPropValuesHelper.getPropValues("dl4j_home.data")
        ).toString();
        parentDir = Paths.get(dataDir,"melanomaChallenge","dataset_unsupervised").toString();
        //parentDir = Paths.get(dataDir,"OilPalm_Images").toString();
        log.info("Folders containing train and test data located \nat: "+parentDir);
    }
    //Note on working with labelled data (supervised learning): org.datavec.api.io.filter.BalancedPathFilter
    //cannot be used! It can only be used with unlabelled data, since PathLabelGenerator must be included in args
    //Thus, input data must be balanced manually. Can do this by InputSplit[]
    //train and test source folders
    public static void setup(int batchSizeArg, int trainPercentage) throws IOException{
        batchSize = batchSizeArg;

        log.info("Getting parent data drive...");
        loadData();

//        Path trainDataPath = Paths.get(parentDir,trainfolder);
//        Path testDataPath = Paths.get(parentDir,testfolder);
//        System.out.println("Location of training data: "+trainDataPath.toString());
//        System.out.println("Location of test data: "+testDataPath.toString());
//
//        trainData = new FileSplit(new File(trainDataPath.toString()));
//        testData = new FileSplit(new File(testDataPath.toString()));


        //The approach below hopes the model learns each image class (melanoma, not_melanoma) on its own.
        //It combines all images into 1 folder, splits into train & test, then iterates
        //Unfortunately, this does not work for our input images
        File parentFile = new File(parentDir);

        //Files in directories under the parent dir that have "allowed extensions" split needs a random number generator for reproducibility when splitting the files into train and test
        FileSplit filesInDir = new FileSplit(parentFile, allowedExtensions, rng);

        //The balanced path filter gives you fine tune control of the min/max cases to load for each class
        //It randomises the input paths, thus the images are not fed according to their sequence in the source folder
        BalancedPathFilter pathFilter = new BalancedPathFilter(rng, allowedExtensions, labelMaker);
        if (trainPercentage >= 100) {
            throw new IllegalArgumentException(
                "Percentage of data set aside for training has to be less than 100%.\n" +
                "Test percentage = 100 - training percentage, has to be greater than 0");
        }

        //Split the image files into train and test
        InputSplit[] filesInDirSplit = filesInDir.sample(pathFilter, trainPercentage, 100-trainPercentage);
        trainData = filesInDirSplit[0];
        testData = filesInDirSplit[1];
    }


    private static DataSetIterator makeIterator(InputSplit split, boolean training) throws IOException{
        //Force-crops input images to same size of 256*256px
        ImageTransform resizeImage = new ResizeImageTransform(224, 224);
        //Lists ImageTransform to be applied and a Double to specify odds that the ImageTransform will be applied
        List<Pair<ImageTransform, Double>> transformPipeline = Arrays.asList(
                new Pair<>(resizeImage, 1.0)
        );
        transform = new PipelineImageTransform(transformPipeline, false);

        //Cannot be used for supervised learning since it won't accept either PascalVOC or YOLO labels
        //This can only work with unsupervised(partially supervised?) learning, since it makes labels based on paths
        ImageRecordReader recordReader = new ImageRecordReader(
            height,width,channels,labelMaker,transform
        );

        //This if-else check ensures transforms are only applied to the InputSplit containing trainData
        if (training == true && transform != null){
            recordReader.initialize(split);
        }else{
            recordReader.initialize(split);
        }
        DataSetIterator iter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numClasses);
        //Apply min-max normalisation as pre-processing
        iter.setPreProcessor(scaler);

        return iter;
    }
    public static DataSetIterator trainIterator() throws Exception{
        return makeIterator(trainData, true);
    }
    public static DataSetIterator testIterator() throws  Exception{
        return makeIterator(testData, false);
    }
}