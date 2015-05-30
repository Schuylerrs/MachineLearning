package NeuralNet;

import java.util.ArrayList;

/**
 *
 * @author Schuyler
 */
public class NeuronLayer 
{
    private final ArrayList<Neuron> neurons = new ArrayList<>();
    private double bias;
    private boolean learning = false;
    
    public NeuronLayer(int numNeurons, int numInputs, float bias)
    {
        for (int i = 0; i < numNeurons; i++)
        {
            neurons.add(new Neuron(numInputs + 1));
        }
    }
    
    public ArrayList<Double> forwardPropigate(ArrayList<Double> input)
    {
        input.add(bias);
        ArrayList<Double> result = new ArrayList<>();
        
        for (int i = 0; i < neurons.size(); i++)
        {
            result.add(neurons.get(i).fire(input));
        }
        
        return result;
    }
}
