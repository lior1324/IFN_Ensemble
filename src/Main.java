import DataSet.DataSet;
import IFN.IFNForestClassifier;
import IFN.InfoFuzzyNetwork;

import java.util.ArrayList;
import java.util.Random;

public class Main {
    public static void main(String[] args) {
        //RunSingleIFNExample();
        //RunIFNForestExample();
        //RunSingleIFNTrainTestExample();
        RunIFNForestTrainTestExample();
    }
    public static void RunSingleIFNExample(){
        String PATH_TO_FILE = "credit_approval\\credit_approval.csv"; // if you want check Liver data use: "Liver\\Liver.csv"
        int NUM_OF_INPUT_ATTRIBUTES = 14; // if you want check Liver data use: 6
        int NUM_OF_DIFFERENT_VALUES_IN_TARGET_ATTRIBUTE = 2; // if you want check Liver data use: 2
        int MAX_DEPTH = -1; //-1 is default -> grow network to max size. enter a number if you want to change
        int MAX_FEATURE = -1; //-1 is default -> use all input attributes as candidates to each split. enter a number if you want to change to randomize number of features to be candidates
        double SIGNIFICANCE = 0.001;
        boolean PREPRUNE = true; //whether to pre prune by G stat test or not when train and build the network.
        boolean PRINT_NET = true; // whether print the network when train it.
        boolean RUN_CONFUSION_METRIX = true;

        DataSet ds = new DataSet(PATH_TO_FILE,NUM_OF_INPUT_ATTRIBUTES,NUM_OF_DIFFERENT_VALUES_IN_TARGET_ATTRIBUTE);
        InfoFuzzyNetwork ifn = new InfoFuzzyNetwork(MAX_DEPTH,MAX_FEATURE,SIGNIFICANCE,PREPRUNE,true);
        ifn.fit(ds,PRINT_NET);
        int[] preds = ifn.predict(ds.X);
        if(RUN_CONFUSION_METRIX){
            DataSet.confusion_metrix(preds,ds.y,NUM_OF_DIFFERENT_VALUES_IN_TARGET_ATTRIBUTE,true);
        }
    }

    public static void RunIFNForestExample(){
        String PATH_TO_FILE = "credit_approval\\credit_approval.csv"; // if you want check Liver data use: "Liver\\Liver.csv"
        int NUM_OF_INPUT_ATTRIBUTES = 14; // if you want check Liver data use: 6
        int NUM_OF_DIFFERENT_VALUES_IN_TARGET_ATTRIBUTE = 2; // if you want check Liver data use: 2
        boolean BOOTSTRAP = true; //use bootstrap sample to train each model
        int N_ESTIMATORS = 10; // number of ifn models in the forest
        int MAX_DEPTH = -1; //-1 is default -> grow network to max size. enter a number if you want to change
        int MAX_FEATURE = 4; //-1 is default -> use all input attributes as candidates to each split. enter a number if you want to change to randomize number of features to be candidates
        double SIGNIFICANCE = 0.001;
        boolean PREPRUNE = true; //whether to pre prune by G stat test or not when train and build the network.
        boolean RUN_CONFUSION_METRIX = true;

        DataSet ds = new DataSet(PATH_TO_FILE,NUM_OF_INPUT_ATTRIBUTES,NUM_OF_DIFFERENT_VALUES_IN_TARGET_ATTRIBUTE);
        IFNForestClassifier ifnForest = new IFNForestClassifier(BOOTSTRAP,MAX_DEPTH,MAX_FEATURE,N_ESTIMATORS,1,SIGNIFICANCE,PREPRUNE,true);
        ifnForest.fit(ds,NUM_OF_DIFFERENT_VALUES_IN_TARGET_ATTRIBUTE);
        int[] preds = ifnForest.predict(ds.X);
        if(RUN_CONFUSION_METRIX){
            DataSet.confusion_metrix(preds,ds.y,NUM_OF_DIFFERENT_VALUES_IN_TARGET_ATTRIBUTE,true);
        }
    }

    public static void RunSingleIFNTrainTestExample(){
        String PATH_TO_FILE_TRAIN = "credit_approval\\train.csv"; // if you want check Liver data use: "Liver\\train.csv"
        String PATH_TO_FILE_TEST = "credit_approval\\test.csv"; // if you want check Liver data use: "Liver\\test.csv"
        int NUM_OF_INPUT_ATTRIBUTES = 14; // if you want check Liver data use: 6
        int NUM_OF_DIFFERENT_VALUES_IN_TARGET_ATTRIBUTE = 2; // if you want check Liver data use: 2
        int MAX_DEPTH = -1; //-1 is default -> grow network to max size. enter a number if you want to change
        int MAX_FEATURE = -1; //-1 is default -> use all input attributes as candidates to each split. enter a number if you want to change to randomize number of features to be candidates
        double SIGNIFICANCE = 0.001;
        boolean PREPRUNE = true; //whether to pre prune by G stat test or not when train and build the network.
        boolean PRINT_NET = true; // whether print the network when train it.
        boolean RUN_CONFUSION_METRIX = true;

        DataSet trainData = new DataSet(PATH_TO_FILE_TRAIN,NUM_OF_INPUT_ATTRIBUTES,NUM_OF_DIFFERENT_VALUES_IN_TARGET_ATTRIBUTE);
        DataSet testData = new DataSet(PATH_TO_FILE_TEST,NUM_OF_INPUT_ATTRIBUTES,NUM_OF_DIFFERENT_VALUES_IN_TARGET_ATTRIBUTE);
        InfoFuzzyNetwork ifn = new InfoFuzzyNetwork(MAX_DEPTH,MAX_FEATURE,SIGNIFICANCE,PREPRUNE,true);
        ifn.fit(trainData,PRINT_NET);
        int[] preds = ifn.predict(testData.X);
        if(RUN_CONFUSION_METRIX){
            DataSet.confusion_metrix(preds,testData.y,NUM_OF_DIFFERENT_VALUES_IN_TARGET_ATTRIBUTE,true);
        }
    }

    public static void RunIFNForestTrainTestExample(){
        String PATH_TO_FILE_TRAIN = "credit_approval\\train.csv"; // if you want check Liver data use: "Liver\\train.csv"
        String PATH_TO_FILE_TEST = "credit_approval\\test.csv"; // if you want check Liver data use: "Liver\\test.csv"
        int NUM_OF_INPUT_ATTRIBUTES = 14; // if you want check Liver data use: 6
        int NUM_OF_DIFFERENT_VALUES_IN_TARGET_ATTRIBUTE = 2; // if you want check Liver data use: 2
        boolean BOOTSTRAP = true; //use bootstrap sample to train each model
        int N_ESTIMATORS = 10; // number of ifn models in the forest
        int MAX_DEPTH = -1; //-1 is default -> grow network to max size. enter a number if you want to change
        int MAX_FEATURE = 4; //-1 is default -> use all input attributes as candidates to each split. enter a number if you want to change to randomize number of features to be candidates
        double SIGNIFICANCE = 0.001;
        boolean PREPRUNE = true; //whether to pre prune by G stat test or not when train and build the network.
        boolean RUN_CONFUSION_METRIX = true;

        DataSet trainData = new DataSet(PATH_TO_FILE_TRAIN,NUM_OF_INPUT_ATTRIBUTES,NUM_OF_DIFFERENT_VALUES_IN_TARGET_ATTRIBUTE);
        DataSet testData = new DataSet(PATH_TO_FILE_TEST,NUM_OF_INPUT_ATTRIBUTES,NUM_OF_DIFFERENT_VALUES_IN_TARGET_ATTRIBUTE);
        IFNForestClassifier ifnForest = new IFNForestClassifier(BOOTSTRAP,MAX_DEPTH,MAX_FEATURE,N_ESTIMATORS,1,SIGNIFICANCE,PREPRUNE,true);
        ifnForest.fit(trainData,NUM_OF_DIFFERENT_VALUES_IN_TARGET_ATTRIBUTE);
        int[] preds = ifnForest.predict(testData.X);
        if(RUN_CONFUSION_METRIX){
            DataSet.confusion_metrix(preds,testData.y,NUM_OF_DIFFERENT_VALUES_IN_TARGET_ATTRIBUTE,true);
        }
    }
}
