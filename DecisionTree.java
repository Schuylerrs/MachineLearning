package machinelearning;

import java.util.Enumeration;
import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

/**
 *
 * @author Schuyler
 */
public class DecisionTree extends Classifier
{
    private Attribute mSplitAttribute = new Attribute("No Data", 0);
    private double mClassGuess = -1;
    private DecisionTree[] mChildren = new DecisionTree[0];
    
    @Override
    public void buildClassifier(Instances data) throws Exception 
    {
        data = new Instances(data);
        
        buildTree(data);
    }
    
    public void drawTree(String prefix)
    {
        if (mChildren.length > 0)
        {
            System.out.print(prefix + mSplitAttribute.name() + "\n");
            for (DecisionTree mChild : mChildren) 
            {
                mChild.drawTree("|  " + prefix);
            }
        }
        else
        {
            System.out.print(prefix + mClassGuess + "\n");
        }
    }
    
  private void buildTree(Instances data) throws Exception 
  {    
    if (data.numInstances() == 0)
    {
        mClassGuess = 0;
        return;
    }
    
    double[] infoGain = new double[data.numAttributes()];
    Enumeration attributes = data.enumerateAttributes();
    
    while (attributes.hasMoreElements()) 
    {
      Attribute att = (Attribute) attributes.nextElement();
      infoGain[att.index()] = computeInfoGain(data, att);
    }
    
    // Find attribute with most information gain
    int maxIndex = 0;
    for (int i = 1; i < infoGain.length; i++)
    {
        double newnumber = infoGain[i];
        if ((newnumber > infoGain[maxIndex]))
            maxIndex = i;   
    }
    mSplitAttribute = data.attribute(maxIndex);
    
    // If there is only one class or only one attribute make a leaf
    // Else build children
    if (data.numClasses() == 1 || infoGain[maxIndex] == 0)
    {
        double[] classes = new double[data.numClasses()];
        
        Enumeration instances = data.enumerateInstances();
        while (instances.hasMoreElements()) 
        {
            Instance inst = (Instance) instances.nextElement();
            classes[(int) inst.classValue()]++;
        }
      
        maxIndex = 0;
        for (int i = 1; i < classes.length; i++)
        {
            double newnumber = classes[i];
            if ((newnumber > classes[maxIndex]))
            {
                maxIndex = i;
            }   
        }
      
        mClassGuess = maxIndex;
    } 
    else 
    {
        // Split the data based on the attribute
        Instances[] splitData = splitData(data, mSplitAttribute);
        // Build a new node for each value
        mChildren = new DecisionTree[mSplitAttribute.numValues()];
        for (int j = 0; j < mSplitAttribute.numValues(); j++) 
        {
            mChildren[j] = new DecisionTree();
            mChildren[j].buildTree(splitData[j]);
        }
    }
  }

    
    @Override
    public double classifyInstance(Instance instance)
    {
        if (mClassGuess != -1) 
        {
            return mClassGuess;
        } 
        else 
        {
            return mChildren[(int) instance.value(mSplitAttribute)].classifyInstance(instance);
        }
    }    
    
  /**
   * Computes information gain for an attribute
   */
  private double computeInfoGain(Instances data, Attribute att) 
    throws Exception 
  {

    double infoGain = computeEntropy(data);
    Instances[] splitData = splitData(data, att);
    for (int j = 0; j < att.numValues(); j++) 
    {
      if (splitData[j].numInstances() > 0) 
      {
        infoGain -= ((double) splitData[j].numInstances() /
                     (double) data.numInstances()) *
          computeEntropy(splitData[j]);
      }
    }
    return infoGain;
  }
   
  /**
   * Computes the entropy
   */
  private double computeEntropy(Instances data) throws Exception {

    double [] classCounts = new double[data.numClasses()];
    Enumeration instEnum = data.enumerateInstances();
    
    while (instEnum.hasMoreElements()) 
    {
      Instance inst = (Instance) instEnum.nextElement();
      classCounts[(int) inst.classValue()]++;
    }
    double entropy = 0;
    
    for (int j = 0; j < data.numClasses(); j++) 
    {
      if (classCounts[j] > 0) 
      {
        entropy -= classCounts[j] * Utils.log2(classCounts[j]);
      }
    }
    entropy /= (double) data.numInstances();
    return entropy + Utils.log2(data.numInstances());
  }
  
  /**
   * Splits a data based on an attribute
   */
  private Instances[] splitData(Instances data, Attribute att)
  {
    int types = att.numValues();    
    Instances newData = data;
    Instances[] splitData = new Instances[types];
    
    for (int j = 0; j < types; j++) 
    {
      splitData[j] = new Instances(newData, newData.numInstances());
    }

    Enumeration instEnum = newData.enumerateInstances();

    while (instEnum.hasMoreElements()) 
    {
      Instance inst = (Instance) instEnum.nextElement();
      splitData[(int) inst.value(att)].add(inst);
    }

    for (Instances splitData1 : splitData) 
    {
        splitData1.compactify();
    }
    return splitData;
  }

    
}
