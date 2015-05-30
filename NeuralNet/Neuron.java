package NeuralNet;

import java.util.ArrayList;
import java.util.Random;

/**
 *
 * @author Schuyler
 */
public class Neuron 
{
    private final ArrayList<Double> weights = new ArrayList<>();
    
    public Neuron(int size)
    {
        Random rnd = new Random(Double.doubleToLongBits(Math.random()));
        
        for (int i = 0; i < size; i++)
            weights.add(rnd.nextDouble() - rnd.nextDouble());
    }
    
    public double fire(ArrayList<Double> inputs)
    {
        float sum = 0;
        float result = 0;
        
        for (int i = 0; i < weights.size(); i++)
        {
            sum += (inputs.get(i) * weights.get(i));
        }
        
        return 1.0 / (1.0 + Math.exp(sum * -1.0));
    }
}
