/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package NeuralNet;

import java.util.ArrayList;
import weka.core.Instance;

/**
 *
 * @author Schuyler
 */
public class NeuralNetwork 
{
    private ArrayList layers;
    private int numAttributes;
    
    public NeuralNetwork(int inputs, ArrayList nodesPerLayer)
    {
        layers = new ArrayList();
        numAttributes = inputs;
        
        int weightsInNext = inputs;
        int numLayers = nodesPerLayer.size();
        
        for(int i = 0; i < numLayers; i++)
        {
            layers.add(new NeuronLayer((int) nodesPerLayer.get(i), weightsInNext, -1));
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
        
        return (double)inputs.get(maxPos);
    }
}
