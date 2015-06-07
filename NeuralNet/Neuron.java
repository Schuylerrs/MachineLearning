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
    private final double learningRate;
    private double lastOutput;
    
    
    public Neuron(int size, double learningSpeed)
    {
        Random rnd = new Random(Double.doubleToLongBits(Math.random()));
        this.learningRate = learningSpeed;
        
        for (int i = 0; i < size; i++)
            weights.add(rnd.nextDouble() - rnd.nextDouble());
        
        System.out.println();
    }
    
    public double fire(ArrayList<Double> inputs)
    {
        double sum = 0;
        double result = 0;
        
        for (int i = 0; i < weights.size(); i++)
        {
            sum += (inputs.get(i) * weights.get(i));
        }
        
        lastOutput = 1.0 / (1.0 + Math.exp(sum * -1.0));
        
        return lastOutput;
    }
    
    public double getWeight(int i)
    {
        return weights.get(i);
    }
    
    public ArrayList<Double> getWeights()
    {
        return weights;
    }
    
    public ArrayList<Double> update(ArrayList<Double> errors)
    {
        double error = getError(errors);
        ArrayList<Double> backPropagation = backPropagateError(error);
        
        updateWeights(error);
        
        return backPropagation;
    }
    
    public ArrayList<Double> update(double target)
    {
        double error = getError(target);
        ArrayList<Double> backPropagation = backPropagateError(error);
        
        updateWeights(error);
        
        return backPropagation;
    }
    
    private void updateWeights(double error)
    {
        double newWeight;
        
        for (int i = 0; i < weights.size(); i++)
        {
            newWeight = weights.get(i) - (learningRate * error * lastOutput);
            weights.set(i, newWeight);
        }
    }
    
    private double getError(double target)
    {
        return lastOutput * (1.0 - lastOutput) * (lastOutput - target);
    }
    
    private double getError(ArrayList<Double> errors)
    {
        double sum = 0;
        
        for (int i = 0; i < errors.size(); i++)
        {
            sum += errors.get(i);
        }
        
        return lastOutput * (1.0 - lastOutput) * sum;
    }
    
    private ArrayList<Double> backPropagateError(double pError)
    {
        ArrayList<Double> error = new ArrayList();
        
        for (int i = 0; i < weights.size(); i++)
        {
            error.add(weights.get(i) * pError);
        }
        
        return error;
    }
}
