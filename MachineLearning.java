package machinelearning;

import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.RemovePercentage;
import java.util.Date;
import java.util.Random;

/**
 * Runs a very basic test using a hard coded classifier
 * @author Schuyler
 */

public class MachineLearning
{
    public static void main(String[] args) throws Exception 
    {
        // Setting up random class and seeding with time
        Date date= new Date();
	Random rand = new Random();
        rand.setSeed(date.getTime());

        // Getting the file set up (I know there is a better way to do this)
        DataSource source = new DataSource("C:\\Users\\Schuyler\\Downloads\\iris.csv");
        Instances data = source.getDataSet();
        data.setClassIndex(data.numAttributes() - 1);
        data.randomize(rand);
 
        // Seting up the remove
        RemovePercentage remove = new RemovePercentage();
        remove.setInputFormat(data);
        remove.setPercentage(30);
        
        // Setting up training data
        Instances trainingData = Filter.useFilter(data, remove);

        // Setting up the Test Data
        remove.setInputFormat(data);
        remove.setInvertSelection(true);
        Instances testData = Filter.useFilter(data, remove);
 
        // Running the test
        HardCodedClassifier classifier = new HardCodedClassifier();
        classifier.buildClassifier(trainingData);
        
        // Setting up evaluation and printing results
        Evaluation evaluation = new Evaluation(trainingData);
        evaluation.evaluateModel(classifier, testData);
        String summary = "Results\n-----------------\nTest Instances: " 
                + testData.numInstances() 
                + "Training Instances: " 
                + trainingData.numInstances() + "\n";
        System.out.println(evaluation.toSummaryString(summary, false));
    }
}
