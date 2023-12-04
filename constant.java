package org.njit.ss4687;

import org.apache.log4j.LogManager;
import org.apache.log4j.Logger;

public class Constants {
    public static final Logger LOGGER = LogManager.getLogger(ModelTrainer.class);

    public static final String TRAINING_DATASET = "data/TrainingDataset.csv";
    public static final String VALIDATION_DATASET = "data/ValidationDataset.csv";
    public static final String MODEL_PATH = "data/TrainingModel";
    public static final String TESTING_DATASET = "data/TestDataset.csv";

    public static final String APP_NAME = "Wine-quality-test";
}