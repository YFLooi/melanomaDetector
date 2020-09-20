package global.melanomaDetector;

import global.melanomaDetector.GetPropValuesHelper;
import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.objdetect.ObjectDetectionRecordReader;
import org.datavec.image.recordreader.objdetect.impl.VocLabelProvider;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.CompositeDataSetPreProcessor;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.dataset.api.preprocessor.classimbalance.UnderSamplingByMaskingPreProcessor;
import org.nd4j.shade.protobuf.compiler.PluginProtos;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
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
    private static final int nChannels = 3;
    private static final int numClasses = 2;
    private static final int height = 256;
    private static final int width = 256;

    private static ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
    private static DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
    private static InputSplit trainData,testData;
    private static int batchSize;

    public static RecordReaderDataSetIterator trainIterator(int batchSize) throws Exception{
        return makeIterator(trainData);
    }
    public static  RecordReaderDataSetIterator testIterator(int batchSize) throws  Exception{
        return makeIterator(testData);
    }
    private static RecordReaderDataSetIterator makeIterator(InputSplit split) throws Exception{
        RecordReaderDataSetIterator iter = new RecordReaderDataSetIterator(
                recordReader,batchSize,1,1,true);

        //Other pre-processors: https://deeplearning4j.org/api/latest/org/nd4j/linalg/dataset/api/preprocessor/package-summary.html
        //CompositeDataSetPreProcessor allows applying multiple preProcesses at one go
        //preProcessors run in order specified from left to right
        DataSetPreProcessor preProcessorList = new CompositeDataSetPreProcessor(
                new ImagePreProcessingScaler(0,1)
        );
        iter.setPreProcessor(preProcessorList);

        return iter;
    }


    //setup() and loadData() prepare data to be loaded into the model
    //Note on working with labelled data (supervised learning): org.datavec.api.io.filter.BalancedPathFilter
    //cannot be used! It can only be used with unlabelled data, sicne PathLabelGenerator must be included in args
    //Thus, input data must be balanced manually, i.e. have same number of images for each class in
    //train and test source folders
    public static void setup() throws IOException{
        log.info("Loading data......");
        loadData();
        trainDir=Paths.get(parentDir,trainfolder);
        testDir=Paths.get(parentDir,testfolder);

        log.info("Path to train images: "+trainDir);
        log.info("Path to test images: "+testDir);

        trainData = new FileSplit(new File(trainDir.toString()), NativeImageLoader.ALLOWED_FORMATS, rng);
        testData = new FileSplit(new File(testDir.toString()),NativeImageLoader.ALLOWED_FORMATS,rng);
    }

    //setup() and loadData() prepare data to be loaded into the YOLO model
    public static void unsupervisedLearningSetup() throws IOException{
        log.info("Loading data......");
        dataDir = System.getProperty("user.home")+"/.deeplearning4j/data/melanomaChallenge/dataset";

        File datasetFolder= new File(dataDir, "/data");
        File labelFolder= new File(dataDir, "/data/annotations");

        //train & test data in same folder. inputSplit splits it into train and test set
        trainDir = Paths.get(datasetFolder.toString());
        testDir = Paths.get(datasetFolder.toString());

        log.info("Source folder located at: " + datasetFolder.toString());
        log.info("Label folder located at: " + labelFolder.toString());

        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        FileSplit fileSplit = new FileSplit(datasetFolder, NativeImageLoader.ALLOWED_FORMATS, rng);
        int numExamples = Math.toIntExact(datasetFolder.length());
        //int numLabels = fileSplit.getRootDir().listFiles(File::isDirectory).length; //This only works if your root is clean: only label subdirs.
        int numLabels = Math.toIntExact(labelFolder.length());
        log.info("Number of images: " + numExamples);
        log.info("Number of image labels: " + numLabels);

        //Usage ref: https://deeplearning4j.konduit.ai/getting-started/cheat-sheet#iterators-build-in-dl-4-j-provided-data
        //Apply BalancedPathFilter here to apply under/over sampling by setting maxPathsPerLabel
        //maxPathsPerLabel = 526, the number of melanoma imgs available in dataSet. We have an excess of non-melanoma imgs after all
        BalancedPathFilter pathFilter = new BalancedPathFilter(rng, labelMaker, numExamples, numLabels, 526);

        double splitTrainTest = 0.2;
        //Params: BalancedPathFilter, %f to train set, %f to test set
        InputSplit[] inputSplit = fileSplit.sample(pathFilter, splitTrainTest, 1 - splitTrainTest);

//        trainData = inputSplit[0];
//        testData = inputSplit[1];
    }

    private static void loadData() throws IOException{
        //dataDir creates a path of "C:\Users\win10AccountName\.deeplearning4j\data"
        dataDir= Paths.get(
                System.getProperty("user.home"),
                GetPropValuesHelper.getPropValues("dl4j_home.data")
        ).toString();
        parentDir = Paths.get(dataDir,"melanomaChallenge","dataset").toString();
        log.info("Folders containing train and test data located \nat: "+parentDir);
    }
}