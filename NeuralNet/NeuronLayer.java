package NeuralNet;

import java.util.ArrayList;

/**
 *
 * @author Schuyler
 */
public class NeuronLayer 
{
    private final ArrayList<Neuron> neurons = new ArrayList<>();
    
    public NeuronLayer(int numNeurons, int numInputs, double learningRate)
    {
        for (int i = 0; i < numNeurons; i++)
        {
            neurons.add(new Neuron(numInputs + 1, learningRate));
        }
    }
    
    public ArrayList<Double> forwardPropigate(ArrayList<Double> input)
    {
        input.add(-1.0);
        ArrayList<Double> result = new ArrayList<>();
        
        for (int i = 0; i < neurons.size(); i++)
        {
            result.add(neurons.get(i).fire(input));
        }
        
        return result;
    }
    
    public int getSize()
    {
        return neurons.size();
    }
    
    public ArrayList<ArrayList<Double>> backPropigate(ArrayList<Double> targets)
    {
        ArrayList<ArrayList<Double>> errors = new ArrayList();
        ArrayList<Double> result = neurons.get(0).update(targets.get(0));
        
        for (int i = 0; i < result.size(); i++)
        {
            ArrayList<Double> row = new ArrayList();
            row.add(result.get(i));
            errors.add(row);
        }
        
        for (int i = 1; i < neurons.size(); i++)
        {
            result = neurons.get(i).update(targets.get(i));
        
            for (int j = 0; j < result.size(); j++)
            {
                errors.get(j).add(result.get(j));
            }
        }
        
        return errors;
    }
    
    public ArrayList<ArrayList<Double>> backPropigateHidden(ArrayList<ArrayList<Double>> pErrors)
    {
        ArrayList<ArrayList<Double>> errors = new ArrayList();
        ArrayList<Double> result = neurons.get(0).update(pErrors.get(0));
        
        for (int i = 0; i < result.size(); i++)
        {
            ArrayList<Double> row = new ArrayList();
            row.add(result.get(i));
            errors.add(row);
        }
        
        for (int i = 1; i < neurons.size(); i++)
        {
            result = neurons.get(i).update(pErrors.get(i));
        
            for (int j = 0; j < result.size(); j++)
            {
                errors.get(j).add(result.get(j));
            }
        }
        
        return errors;
    }
}
