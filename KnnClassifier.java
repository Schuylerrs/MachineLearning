/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package machinelearning;

import java.util.Map;
import java.util.TreeMap;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author Schuyler
 */
public class KnnClassifier extends Classifier
{
    private Instances trainingData;
    
    @Override
    public void buildClassifier(Instances data) throws Exception 
    {
        trainingData = data;
    }
    
    @Override
    public double classifyInstance(Instance instance)
    {
        return this.classifyInstance(instance, 3);
    }
    
    public double classifyInstance(Instance instance, int k)
    {
        int size = trainingData.numInstances();
        int attributes = trainingData.numAttributes() - 1;
        float dist;
        Map<Float, Instance> neighbors = new TreeMap<>();
        
        Instance test;

        for (int i = 0; i < size; i++)
        {
            dist = 0;
            test = trainingData.instance(i);
            
            for (int j = 0; j < attributes; j++)
            {   
                if (test.attribute(j).isNominal())
                {
                    if (instance.attribute(j).equals(test.attribute(j)))
                        dist++;
                }
                else
                    dist += Math.abs(test.value(test.attribute(j)) - instance.value(test.attribute(j)));
            }

            neighbors.put(dist, test);
        }
        
        return findMostCommon(neighbors, k);
    }
    
    private double findMostCommon(Map<Float, Instance> neighbors, int k)
    {
        int count = 0;
        int[] classCount = new int[trainingData.numClasses()];
        double classGuess = -1;
        int classSize = 0;
        int index = trainingData.firstInstance().classIndex();
        
        for (Map.Entry<Float,Instance> entry : neighbors.entrySet()) 
        {
            if (count == k) 
                break;

            classCount[(int)entry.getValue().value(index)]++;
            
            if (classCount[(int)entry.getValue().value(index)] > classSize)
            {
                classGuess = entry.getValue().value(index);
                classSize = classCount[(int)entry.getValue().value(index)];
            }

            count++;
        }
        
        return classGuess;
    }
}
