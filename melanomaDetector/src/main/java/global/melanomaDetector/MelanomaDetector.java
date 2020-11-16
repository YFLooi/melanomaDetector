package global.melanomaDetector;

import javafx.scene.Parent;
import org.bytedeco.javacv.CanvasFrame;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.FrameGrabber;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Point;
import org.bytedeco.opencv.opencv_core.Scalar;
import org.bytedeco.opencv.opencv_core.Size;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.recordreader.objdetect.impl.VocLabelProvider;
import org.datavec.image.transform.ColorConversionTransform;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.evaluation.EvaluationTools;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.objdetect.Yolo2OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.layers.objdetect.DetectedObject;
import org.deeplearning4j.nn.layers.objdetect.YoloUtils;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.VGG16;
import org.deeplearning4j.zoo.model.YOLO2;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.evaluation.classification.ROC;
import org.nd4j.evaluation.classification.ROCBinary;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.function.Function;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.awt.event.KeyEvent;
import java.io.File;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.List;

import static org.bytedeco.opencv.global.opencv_core.CV_8U;
import static org.bytedeco.opencv.global.opencv_core.flip;
import static org.bytedeco.opencv.global.opencv_imgproc.*;
import static org.bytedeco.opencv.helper.opencv_core.RGB;

public class MelanomaDetector {
    //***Set model parameters***
    private static final Logger log = LoggerFactory.getLogger(MelanomaDetector.class);
    private static int seed = 123;

    //***Set model run parameters***
    private static int batchSize = 8;
    private static int trainPercentage = 80;
    //108 iterations per epoch
    //With current setting, 2 epochs gives best results. More epochs do nothing to improve loss
    private static int nEpochs = 2;
    private static double learningRate = 1e-4;

    //2 possible outputs: melanoma and not_melanoma
    //If this changes, adjust nOut of conv2d_23 at getComputationGraph()
    //This ensures output CNN array dimensions matches that of input at conv2d_1
    private static int nClasses = 2;

    //***Set modelFilename and variable for ComputationGraph***
    //Refers to C:\devBox\melanomaDetector\generated-models
    private static File modelFilename = new File(
        System.getProperty("user.dir"),
        "generated-models/melanomaDetector_vgg2Class.zip");
    private static ComputationGraph model;


    public static void main(String[] args) throws Exception{
        SkinDatasetIterator.setup(batchSize, trainPercentage);
        DataSetIterator trainIter = SkinDatasetIterator.trainIterator();
        DataSetIterator testIter = SkinDatasetIterator.testIterator();

        //Evaluation eval = model.evaluate(testIter);
        Evaluation eval = new Evaluation();

        //If model does not exist, train the model, else directly go to model evaluation and then run real time object detection inference.
        if (modelFilename.exists()) {
            //STEP 2 : Load trained model from previous execution
            Nd4j.getRandom().setSeed(seed);
            log.info("Load model...");
            model = ModelSerializer.restoreComputationGraph(modelFilename);
        } else {
            Nd4j.getRandom().setSeed(seed);

            //STEP 2 : Train the model using Transfer Learning
            //STEP 2.1: Transfer Learning steps - Load YOLOv2 prebuilt model.
            ZooModel zooModel = VGG16.builder().build();
            ComputationGraph vgg16 = (ComputationGraph) zooModel.initPretrained();

            //STEP 2.2: Transfer Learning steps - Model Configurations.
            FineTuneConfiguration fineTuneConf = getFineTuneConfiguration();

            //STEP 2.3: Transfer Learning steps - Modify prebuilt model's architecture
            log.info("Original model setup:");
            log.info(vgg16.summary());

            model = getComputationGraph(vgg16, fineTuneConf);
                log.info("Modified model setup:");
                log.info(model.summary(InputType.convolutional(
                224, 224, nClasses)));


            //STEP 2.4: Training and Save model.
            UIServer server = UIServer.getInstance();
            StatsStorage statsStorage = new InMemoryStatsStorage();

            //StatsStorage statsStorage = new FileStatsStorage(new File(System.getProperty("java.io.tmpdir"), "ui-stats.dl4j"));
            server.attach(statsStorage);
            model.setListeners(
                new StatsListener(statsStorage),
                //Prints the message "Score at iteration x is xxx" every 2 iterations
                new ScoreIterationListener(2)
            );

            log.info("Train model...");
            int iter = 0;
            for (int i = 0; i < nEpochs; i++) {
                trainIter.reset();
                while (trainIter.hasNext()) {
                    model.fit(trainIter.next());

                    //Does a test every even xth iter
                    if (iter > 0 && iter % 51 == 0) {
                        log.info("Evaluating model at iter "+iter +" ....");
                        eval = model.evaluate(testIter);
                        log.info(eval.stats());
                        testIter.reset();
                    }
                    ++iter;
                }
                log.info("*** Completed epoch {} ***", i);
            }

            ModelSerializer.writeModel(model, modelFilename, true);
                System.out.println("Model saved.");
        }

        //Calculates confusion matrix (evaluation) and ROC curve
        ROC roc = new ROC(1);
        log.info("Commencing ROC...");
        model.doEvaluation(testIter, eval, roc);
        double aucObtained = roc.calculateAUC();

        log.info(eval.stats());
        log.info("AUC for the ROC: "+aucObtained);

        //Sets save directory for exported ROC chart
        File rocFilepath = new File(System.getProperty("user.dir"),"/ROC_Curves/rocCurve.html");
        EvaluationTools.exportRocChartsToHtmlFile(roc, rocFilepath);
        log.info("ROC curve html file stored at"+rocFilepath);

        //STEP 3: Evaluate the model's accuracy by using the test iterator.
        //OfflineValidationWithTestDataset(testIter);

        //STEP 4: Inference the model and process the webcam stream and make predictions.
        //webcamDetection();
    }


    private static ComputationGraph getComputationGraph(ComputationGraph pretrained, FineTuneConfiguration fineTuneConf) {
        return new TransferLearning.GraphBuilder(pretrained)
            .fineTuneConfiguration(fineTuneConf)
            .removeVertexKeepConnections("predictions")
            .addLayer("predictions",
                //XENT loss function + Sigmoid activation @output layer for multi-label binary classifcation
                //If num output lcasses >2, use MCXENT + Softmax instead
                new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                    .nIn(4096)
                    .nOut(nClasses)
                    .activation(Activation.SOFTMAX)
                    .build(),
                "fc2")
            .build();
    }
    private static FineTuneConfiguration getFineTuneConfiguration() {
        return new FineTuneConfiguration.Builder()
        .seed(seed)
        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
        .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
        .gradientNormalizationThreshold(1.0)
        .updater(new Adam.Builder().learningRate(learningRate).build())
        .l2(0.00001)
        .activation(Activation.RELU)
        .trainingWorkspaceMode(WorkspaceMode.ENABLED)
        .inferenceWorkspaceMode(WorkspaceMode.ENABLED)
        .build();
    }
}

