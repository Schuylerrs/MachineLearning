/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package NeuralNet;

import java.util.ArrayList;
import java.util.Random;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author Schuyler
 */
public class NeuralNetwork 
{
    private final ArrayList layers;
    private final int numAttributes;
    private final Random rand;
    
    public NeuralNetwork(int inputs, ArrayList nodesPerLayer)
    {
        layers = new ArrayList();
        numAttributes = inputs;
        rand = new Random();
        
        int weightsInNext = inputs;
        int numLayers = nodesPerLayer.size();
        
        for(int i = 0; i < numLayers; i++)
        {
            layers.add(new NeuronLayer((int) nodesPerLayer.get(i), weightsInNext, .1));
            weightsInNext = (int) nodesPerLayer.get(i);
        }
    }
    
    public double passThrough(Instance instance)
    {
        ArrayList inputs = new ArrayList();
        
        for (int i = 0; i < numAttributes; i++)
        {
            inputs.add(instance.value(i));
        }
        
        NeuronLayer temp;
        
        for (int i = 0; i < layers.size(); i++)
        {
            temp = (NeuronLayer) layers.get(i);
            inputs = temp.forwardPropigate(inputs);
        }
        
        int maxPos = 0;
        
        for (int i = 0; i < inputs.size(); i++)
        {
            if ((double)inputs.get(i) > (double)inputs.get(maxPos))
            {
                maxPos = i;
            }
        }
        
        return (double) maxPos;
    }
    
    public double train(Instances data)
    {
        double correct = 0.0;
        double count = 0;
        double guess;
       
        data.randomize(rand);
        for (int i = 0; i < data.numInstances(); i++)
        {
            guess = passThrough(data.instance(i));
            
            if (guess == data.instance(i).classValue())
            {
                correct++;
            }
            count++; 
            
            backPropigate(data.instance(i), data.numClasses());
        }
        
        return correct / count;
    }
    
    private void backPropigate(Instance test, int numClasses)
    {
        ArrayList<Double> targets = new ArrayList();
        double classValue = test.classValue();
        
        // build the targets list
        for (double i = 0; i < numClasses; i++)
        {
            if (i == classValue)
            {
                targets.add(1.0);
            }
            else
            {
                targets.add(0.0);
            }
        }
        
        NeuronLayer layer = (NeuronLayer) layers.get(layers.size() - 1);
        ArrayList<ArrayList<Double>> error = layer.backPropigate(targets);
        
        // iterate backwards starting with the second to last layer
        for (int i = layers.size() - 2; i >= 0; i--)
        {
            layer = (NeuronLayer) layers.get(i);
            error = layer.backPropigateHidden(error);
        }
    }
}
