package org.njit.ss4687;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.evaluation.MulticlassClassificationSummary;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import java.io.IOException;
import java.io.Serializable;

import static org.njit.vb53.Constants.*;

public class ModelTrainer implements Serializable {

    private static final long serialVersionUID = 1L;

    private static final Logger logger = Logger.getLogger(ModelTrainer.class);

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

        File trainingFile = new File(TRAINING_DATASET);
        boolean exists = trainingFile.exists();

        if (exists) {
            ModelTrainer trainer = new ModelTrainer();
            trainer.trainLogisticRegression(spark);
        } else {
            System.out.println("TrainingDataset.csv doesn't exist");
        }
    }

    public void trainLogisticRegression(SparkSession spark) {
        System.out.println();
        Dataset<Row> labeledFeatureDf = getDataFrame(spark, true, TRAINING_DATASET).cache();
        LogisticRegression logisticRegression = new LogisticRegression().setMaxIter(100).setRegParam(0.0);

        Pipeline pipeline = new Pipeline();
        pipeline.setStages(new PipelineStage[]{logisticRegression});

        PipelineModel model = pipeline.fit(labeledFeatureDf);

        LogisticRegressionModel lrModel = (LogisticRegressionModel) (model.stages()[0]);
        LogisticRegressionSummary trainingSummary = lrModel.summary();
        double accuracy = trainingSummary.accuracy();
        double fMeasure = trainingSummary.weightedFMeasure();

        System.out.println();
        System.out.println("Training DataSet Metrics ");
        System.out.println("Accuracy: " + accuracy);
        System.out.println("F-measure: " + fMeasure);

        Dataset<Row> validationDf = getDataFrame(spark, true, VALIDATION_DATASET).cache();
        Dataset<Row> results = model.transform(validationDf);

        System.out.println("\nValidation Training Set Metrics");
        results.select("features", "label", "prediction").show(5, false);
        printMetrics(results);

        try {
            model.write().overwrite().save(MODEL_PATH);
        } catch (IOException e) {
            logger.error(e);
        }
    }

    public void printMetrics(Dataset<Row> predictions) {
        System.out.println();
        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator();
        evaluator.setMetricName("accuracy");
        System.out.println("The accuracy of the model is " + evaluator.evaluate(predictions));
        evaluator.setMetricName("f1");
        double f1 = evaluator.evaluate(predictions);
        System.out.println("F1: " + f1);
    }

    public Dataset<Row> getDataFrame(SparkSession spark, boolean transform, String name) {
        Dataset<Row> validationDf = spark.read().format("csv").option("header", "true")
                .option("multiline", true).option("sep", ";").option("quote", "\"")
                .option("dateFormat", "M/d/y").option("inferSchema", true).load(name);

        validationDf = validationDf.withColumnRenamed("fixed acidity", "fixed_acidity")
                .withColumnRenamed("volatile acidity", "volatile_acidity")
                .withColumnRenamed("citric acid", "citric_acid")
                .withColumnRenamed("residual sugar", "residual_sugar")
                .withColumnRenamed("chlorides", "chlorides")
                .withColumnRenamed("free sulfur dioxide", "free_sulfur_dioxide")
                .withColumnRenamed("total sulfur dioxide", "total_sulfur_dioxide")
                .withColumnRenamed("density", "density").withColumnRenamed("pH", "pH")
                .withColumnRenamed("sulphates", "sulphates").withColumnRenamed("alcohol", "alcohol")
                .withColumnRenamed("quality", "label");

        validationDf.show(5);

        Dataset<Row> labeledFeatureDf = validationDf.select("label", "alcohol", "sulphates", "pH",
                "density", "free_sulfur_dioxide", "total_sulfur_dioxide", "chlorides", "residual_sugar",
                "citric_acid", "volatile_acidity", "fixed_acidity");

        labeledFeatureDf = labeledFeatureDf.na().drop().cache();

        VectorAssembler assembler = new VectorAssembler().setInputCols(
                new String[]{"alcohol", "sulphates", "pH", "density",
                        "free_sulfur_dioxide", "total_sulfur_dioxide", "chlorides", "residual_sugar",
                        "citric_acid", "volatile_acidity", "fixed_acidity"}).setOutputCol("features");

        if (transform)
            labeledFeatureDf = assembler.transform(labeledFeatureDf).select("label", "features");

        return labeledFeatureDf;
    }
}