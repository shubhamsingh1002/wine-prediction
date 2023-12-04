package org.njit.ss4687;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import java.io.File;

import static org.njit.vb53.Constants.*;

public class WineQualityPredictor {

    public static void main(String[] args) {
        Logger.getLogger("org").setLevel(Level.ERROR);
        Logger.getLogger("akka").setLevel(Level.ERROR);
        Logger.getLogger("breeze.optimize").setLevel(Level.ERROR);
        Logger.getLogger("com.amazonaws.auth").setLevel(Level.DEBUG);
        Logger.getLogger("com.github").setLevel(Level.ERROR);

        SparkSession spark = SparkSession.builder()
                .appName(APP_NAME)
                .master("local[*]")
                .config("spark.executor.memory", "2147480000")
                .config("spark.driver.memory", "2147480000")
                .config("spark.testing.memory", "2147480000")
                .getOrCreate();

        File testingFile = new File(TESTING_DATASET);
        boolean testingFileExists = testingFile.exists();

        if (testingFileExists) {
            WineQualityPredictor predictor = new WineQualityPredictor();
            predictor.runLogisticRegression(spark);
        } else {
            System.out.print("TestDataset.csv doesn't exist. Please provide the test file path using -v.\n" +
                    "Example: docker run -v [local_testfile_directory:/data] nieldeokar/wine-prediction-mvn:1.0 /TestDataset.csv\n");
        }
    }

    public void runLogisticRegression(SparkSession spark) {
        System.out.println("Testing DataSet Metrics\n");
        PipelineModel pipelineModel = PipelineModel.load(MODEL_PATH);
        Dataset<Row> testDf = getDataFrame(spark, true, TESTING_DATASET).cache();
        Dataset<Row> predictionDF = pipelineModel.transform(testDf).cache();
        predictionDF.select("features", "label", "prediction").show(5, false);
        printMetrics(predictionDF);
    }

    public Dataset<Row> getDataFrame(SparkSession spark, boolean transform, String name) {
        Dataset<Row> validationDf = spark.read().format("csv").option("header", "true")
                .option("multiline", true).option("sep", ";").option("quote", "\"")
                .option("dateFormat", "M/d/y").option("inferSchema", true).load(name);

        Dataset<Row> labeledFeatureDf = validationDf.withColumnRenamed("quality", "label")
                .select("label", "alcohol", "sulphates", "pH", "density", "free sulfur dioxide",
                        "total sulfur dioxide", "chlorides", "residual sugar", "citric acid",
                        "volatile acidity", "fixed acidity");

        labeledFeatureDf = labeledFeatureDf.na().drop().cache();

        VectorAssembler assembler = new VectorAssembler().setInputCols(
                new String[]{"alcohol", "sulphates", "pH", "density",
                        "free sulfur dioxide", "total sulfur dioxide", "chlorides", "residual sugar",
                        "citric acid", "volatile acidity", "fixed acidity"}).setOutputCol("features");

        if (transform)
            labeledFeatureDf = assembler.transform(labeledFeatureDf).select("label", "features");

        return labeledFeatureDf;
    }

    public void printMetrics(Dataset<Row> predictions) {
        System.out.println();
        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator();
        evaluator.setMetricName("accuracy");
        System.out.println("The accuracy of the model is " + evaluator.evaluate(predictions));

        evaluator.setMetricName("f1");
        double f1 = evaluator.evaluate(predictions);

        System.out.println("F1: " + f1);
    }
}