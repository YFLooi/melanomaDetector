package global.skymind;

import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Point;
import org.bytedeco.opencv.opencv_core.Scalar;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.layers.objdetect.DetectedObject;
import org.deeplearning4j.nn.layers.objdetect.YoloUtils;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.YOLO2;
import org.deeplearning4j.zoo.util.darknet.COCOLabels;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;


import java.io.File;
import java.util.List;

import static org.bytedeco.opencv.global.opencv_highgui.*;
import static org.bytedeco.opencv.global.opencv_imgcodecs.imread;
import static org.bytedeco.opencv.global.opencv_imgproc.*;

public class MelanomaDataSetIterator {
    //     STEP 1: Fix the config for YOLO
    //Splits image into imaginary grid lines.
    //Grid too small will fail to detect, grid too large will not fit in image
    private static final int gridWidth = 28; //About 3.1746% of yolowidth and height
    private static final int gridHeight = 28;
    //Sets threshold of detection. 0.9 = Accepts only objects of >90% confidence
    //Surprisingly, even <50% can return correct results!!
    //Training on higher threshold is useful to adjust gridWidth, gridHeight to set accurate bounding boxes
    private static double detectionThreshold = 0.28;

    //Resizes input image. Note that dimensions must be divisible by grid size (gridWidth, gridHeight) specified!
    //Simplest is to make yolowidth = yoloheight, for a square
    private static final int yolowidth = 882;
    private static final int yoloheight = 882;

    public static void main(String[] args) throws Exception {

//      STEP 2: Enter the PATH to your test image
//        String testImagePATH = "C:\\Users\\deSni\\Downloads\\bb.jpg";
//        File file = new File(testImagePATH);

        //Gets file from dl4j-labs/src/main/resources into dl4j-labs/target/classes
        File newFile = new ClassPathResource("/bb.jpg").getFile();
        String testImagePATH = newFile.getAbsolutePath();

        System.out.println(String.format("You are using this image file located at %s", testImagePATH));
        File file = new File(testImagePATH);
        COCOLabels labels = new COCOLabels();

//      STEP 3: Set output number of classes
        ZooModel yolo2 = YOLO2.builder().build();
        ComputationGraph model = (ComputationGraph) yolo2.initPretrained();
        NativeImageLoader nil = new NativeImageLoader(yolowidth, yoloheight, 3);
        ImagePreProcessingScaler scaler = new ImagePreProcessingScaler(0, 1);
        INDArray image = nil.asMatrix(file);
        scaler.transform(image);

        Mat opencvMat = imread(testImagePATH);
        int w = opencvMat.cols();
        int h = opencvMat.rows();

        INDArray outputs = model.outputSingle(image);
        List<DetectedObject> objs = YoloUtils.getPredictedObjects(Nd4j.create(((YOLO2) yolo2).getPriorBoxes()), outputs, detectionThreshold, 0.4);

        //Draws rectangle for detected objects. Labels obtained from coco labels
        for (DetectedObject obj : objs) {
            double[] xy1 = obj.getTopLeftXY();
            double[] xy2 = obj.getBottomRightXY();
            String label = labels.getLabel(obj.getPredictedClass());
            int x1 = (int) Math.round(w * xy1[0] / gridWidth);
            int y1 = (int) Math.round(h * xy1[1] / gridHeight);
            int x2 = (int) Math.round(w * xy2[0] / gridWidth);
            int y2 = (int) Math.round(h * xy2[1] / gridHeight);
            rectangle(opencvMat, new Point(x1, y1), new Point(x2, y2), Scalar.RED, 2, 0, 0);
            putText(opencvMat, label, new Point(x1 + 2, y2 - 2), FONT_HERSHEY_DUPLEX, 1, Scalar.GREEN);
        }
        imshow("Input Image", opencvMat);

        //        Press "Esc" to close window
        if (waitKey(0) == 27) {
            destroyAllWindows();
        }
    }
}
