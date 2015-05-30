/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package NeuralNet;

import java.util.ArrayList;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author Schuyler
 */
public class NeuralNetworkClassifier 
extends Classifier
{
    private NeuralNetwork neuralNet;
    
    @Override
    public void buildClassifier(Instances data) 
    {
        int attributes = data.numAttributes();
        ArrayList nodesPerLayer = new ArrayList();
        nodesPerLayer.add(attributes + 2);
        
        buildClassifier(data, 1, nodesPerLayer);
    }
    
    public void buildClassifier(Instances data, int hiddenLayers, ArrayList nodesPerLayer) 
    {        
        int outputLayer = data.numClasses();
        nodesPerLayer.add(outputLayer);
        
        int inputs = data.numAttributes() - 1;
        
        neuralNet = new NeuralNetwork(hiddenLayers + 1, nodesPerLayer);
        
        int done = 0;
    }
    
    @Override
    public double classifyInstance(Instance instance)
    {
        return neuralNet.passThrough(instance);
    }
}
