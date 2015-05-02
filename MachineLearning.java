package machinelearning;

import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.RemovePercentage;
import java.util.Date;
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;
import weka.classifiers.Classifier;
import weka.classifiers.lazy.IBk;
import weka.filters.unsupervised.attribute.Standardize;

/**
 * Runs a very basic test using a hard coded classifier
 * @author Schuyler
 */
public class MachineLearning
{
    private final Instances trainingData;
    private final Instances testData;
    
    public MachineLearning(String dataPath) throws Exception
    {
        // Setting up random class and seeding with time
        Date date= new Date();
	Random rand = new Random();
        rand.setSeed(date.getTime());
        
        DataSource source = new DataSource(dataPath);
        Instances data = source.getDataSet();
        Standardize norm = new Standardize();
        norm.setInputFormat(data);
        data = Filter.useFilter(data, norm);
        data.setClassIndex(data.numAttributes() - 1);
        data.randomize(rand);
        
        // Seting up the remove
        RemovePercentage remove = new RemovePercentage();
        remove.setInputFormat(data);
        remove.setPercentage(30);
        
        // Setting up training data
        trainingData = Filter.useFilter(data, remove);
        
        // Setting up the Test Data
        remove.setInputFormat(data);
        remove.setInvertSelection(true);
        testData = Filter.useFilter(data, remove);
    }
    
    public static void main(String[] args) throws Exception 
    {
        //MachineLearning test = new MachineLearning("C:\\Users\\Schuyler\\Downloads\\iris.csv");
        MachineLearning test = new MachineLearning("C:\\Users\\Schuyler\\Downloads\\car.csv");

        test.theirKnnTest(3);
        test.myKnnTest(3);
        //test.hardCodeTest();
       
    }
    
    private void myKnnTest(int k) throws Exception
    {
        KnnClassifier knn = new KnnClassifier();
        knn.buildClassifier(testData);

        evaluate(knn);
    }
    
    private void theirKnnTest(int k) throws Exception
    {
        int index = trainingData.firstInstance().classIndex();
        IBk knn = new IBk(k);
        knn.buildClassifier(trainingData);

        evaluate(knn);
    }
    
    private void hardCodeTest() throws Exception
    {  
        // Running the test
        HardCodedClassifier classifier = new HardCodedClassifier();
        classifier.buildClassifier(trainingData);
        
        evaluate(classifier);
    }
    
    // A standard format for evaluating classifiers
    private void evaluate(Classifier classifier)
    {
        try 
        {
            Evaluation evaluation = new Evaluation(trainingData);
            evaluation.evaluateModel(classifier, testData);
            String summary = "Results\n-----------------\nTest Instances: "
                    + testData.numInstances()
                    + "Training Instances: "
                    + trainingData.numInstances() + "\n";
            System.out.println(evaluation.toSummaryString(summary, false));
        } 
        catch (Exception ex) 
        {
            Logger.getLogger(MachineLearning.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
}
