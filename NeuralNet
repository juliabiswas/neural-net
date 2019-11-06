import java.io.File;
import java.util.Scanner;

/**
 * A multilayer perceptron network that can take in a single set of inputs and produce an output (of one or
 * more values) or train the network on a set of input sets/true values (test cases) and minimize the error
 * using gradient descent.
 *
 * The class contains the following:
 *   a constructor that takes in no parameters but handles all the user-specification retrieval;
 *   a main method which oversees the program;
 *   a function that computes the value of the activation function (sigmoid) given a value;
 *   a function that computes the value of the derivative of the activation function (sigmoid) given a value;
 *   a function that determines whether the network is done training or not;
 *   a function that trains the network; and
 *   a function that runs the network.
 *
 * The network is fully configurable by the user at runtime.
 *
 * @author  Julia Biswas
 * @version October 18, 2019
 */

public class NeuralNet
{
   private int [] nodes; //number of nodes in each layer, the configuration
   private int maxNodes;
   private int testCases; //the number of test cases

   private double [][] inputs;
   private double [][][] weights;
   private double [][] activations;
   private double [][] t; //true (expected) values
   private double [][] f; //the calculated output values

   private double iterations;
   private double lambda;
   private double lambdaChange;
   private double lambdaMin;
   private double lambdaMax;
   private double minRandomWeight;
   private double maxRandomWeight;
   private double desiredError;

   private String inputBMP;
   private String outputBMP;
   private String inputActivation;
   private String outputActivation;

   /*
    * Constructs the neural network using user inputs at runtime.
    */
   public NeuralNet()
   {
      Scanner scan = new Scanner(System.in);
      System.out.println("Please enter the path of the file with the configuration.");
      String path = scan.next();

      try
      {
         Scanner scanFile = new Scanner(new File(path));
         String trainOrTest = scanFile.next();

         //configuration
         int inputNodes = scanFile.nextInt();
         int hiddenLayers = scanFile.nextInt();
         nodes = new int[hiddenLayers + 2];
         nodes[0] = inputNodes;

         for (int n = 1; n <= hiddenLayers; n++)
            nodes[n] = scanFile.nextInt();

         nodes[nodes.length-1] = scanFile.nextInt();

         //computing max number of nodes
         int max = 0; //index
         for (int n = 1; n < nodes.length; n++)
            if (nodes[n] > nodes[max])
               max = n;
         maxNodes = nodes[max];

         weights = new double[nodes.length-1][maxNodes][maxNodes];

         //training or testing
         if (trainOrTest.equals("train"))
         {
            testCases = scanFile.nextInt();
            inputs = new double[testCases][nodes[0]];
            t = new double[testCases][nodes[nodes.length-1]];
            f = new double[testCases][nodes[nodes.length-1]];

            for (int testCase = 0; testCase < testCases; testCase++)
            {
               for (int j = 0; j < inputs[testCase].length; j++)
                  inputs[testCase][j] = scanFile.nextDouble();

               for (int i = 0; i < t[testCase].length; i++)
                  t[testCase][i] = scanFile.nextDouble();
            }//for (int testCase = 0; testCase < testCases; testCase++)

            activations = new double[nodes.length][maxNodes];

            for (int index = 0; index < nodes[0]; index++)
               activations[0][index] = inputs[0][index]; //putting the first input activations
                                                         //into the activations array
            iterations = scanFile.nextDouble();
            lambda = scanFile.nextDouble();
            lambdaChange = scanFile.nextDouble();
            lambdaMin = scanFile.nextDouble();
            lambdaMax = scanFile.nextDouble();
            desiredError = scanFile.nextDouble();

            String randOrSpec = scanFile.next(); //randomized or specified weights

            if (randOrSpec.equals("random"))
            {
               minRandomWeight = scanFile.nextDouble();
               maxRandomWeight = scanFile.nextDouble();

               for (int m = 0; m < nodes.length-1; m++)
               {
                  int maxFrom = nodes[m];

                  if (m > 0)
                     maxFrom = nodes[m-1]; //the "from" nodes are input nodes for first layer of weights

                  for (int from = 0; from < maxFrom; from++)
                  {
                     int maxTo = nodes[nodes.length-1];

                     if (m < nodes.length-2)
                        maxTo = nodes[m+1]; //the "to" nodes are output nodes for last layer of weights

                     for (int to = 0; to < maxTo; to++)
                        weights[m][from][to] = (Math.random()*(maxRandomWeight-minRandomWeight)) + minRandomWeight;
                  }//for (int from = 0; from < maxFrom; from++)
               }//for (int m = 0; m < nodes.length-1; m++)
            }//if (randOrSpec.equals("random"))

            else if (randOrSpec.equals("specified"))
            {
               for (int m = 0; m < nodes.length-1; m++)
               {
                  int maxFrom = nodes[m];

                  if (m > 0)
                     maxFrom = nodes[m-1]; //the "from" nodes are input nodes for first layer of weights

                  for (int from = 0; from < maxFrom; from++)
                  {
                     int maxTo = nodes[nodes.length-1];

                     if (m < nodes.length-2)
                        maxTo = nodes[m+1]; //the "to" nodes are output nodes for last layer of weights

                     for (int to = 0; to < maxTo; to++)
                        weights[m][from][to] = scanFile.nextDouble();
                  }//for (int from = 0; from < maxFrom; from++)
               }//for (int m = 0; m < nodes.length-1; m++)
            }//else if (randOrSpec.equals("specified"))

            trainModel();
         }//if (trainOrTest.equals("train"))

         else if (trainOrTest.equals("test"))
         {
            inputs = new double[1][nodes[0]];
            t = new double[1][nodes[nodes.length-1]];
            f = new double[1][nodes[nodes.length-1]];

            for (int j = 0; j < nodes[0]; j++)
               inputs[0][j] = scanFile.nextDouble();

            for (int i = 0; i < nodes[nodes.length-1]; i++)
               t[0][i] = scanFile.nextDouble();

            for (int m = 0; m < nodes.length-1; m++)
            {
               int maxFrom = nodes[m];

               if (m > 0)
                  maxFrom = nodes[m-1]; //the "from" nodes are input nodes for first layer of weights

               for (int from = 0; from < maxFrom; from++)
               {
                  int maxTo = nodes[nodes.length-1];

                  if (m < nodes.length-2)
                     maxTo = nodes[m+1]; //the "to" nodes are output nodes for last layer of weights

                  for (int to = 0; to < maxTo; to++)
                     weights[m][from][to] = scanFile.nextDouble();
               }//for (int from = 0; from < maxFrom; from++)
            }//for (int m = 0; m < nodes.length-1; m++)

            activations = new double[nodes.length][maxNodes];

            for (int index = 0; index < nodes[0]; index++)
               activations[0][index] = inputs[0][index]; //putting the first input activations
                                                         //into the activations array
            testModel();
            double[] output = activations[activations.length-1];

            System.out.println("\n" + "output: ");

            for (int i = 0; i < nodes[nodes.length-1]; i++)
               System.out.println(output[i]);
         }//else if (trainOrTest.equals("test"))
      }//try

      catch (Exception e)
      {
         System.out.println("Exception occurred trying to read " + path);
      }
   }//public NeuralNet()

   /*
    * Computes the value of the activation function (sigmoid).
    *
    * @param value            the value to plug into the function
    *                         (it should be a dot product computed earlier)
    *
    * @return                 the value of the activation function (sigmoid)
    */
   public double activation(double value)
   {
      return 1.0/(1.0+Math.exp(-value));
   }//public double activation(double value)

   /*
    * Computes the value of the derivative of the activation
    * function (sigmoid).
    *
    * @param value            the value to plug into the function
    *
    * @return                 the value of the derivative of the activation
    *                         function (sigmoid)
    */
   public double activationPrime(double value)
   {
      return activation(value)*(1-activation(value));
   }//public double activationPrime(double value)


   /*
    * Determines whether the training is done or not.
    *
    * The training terminates if the number of iterations
    * has exceeded the max iterations (specified by the user)
    * or if the current error is equal to or better (lower) than
    * the desired error.
    *
    * @param iteration         the current iteration
    * @param error             the current error
    *
    * @return                  true if the training is done; otherwise,
    *                          false
    */
   public boolean isDone(int iteration, double error)
   {
      if (iteration >= iterations)
         return true;

      else if (error <= desiredError)
         return true;

      return false;
   }//public boolean isDone(int iteration, double error)

   /*
    * Trains the model by starting with random weights
    * and updates them for a user-specified number of
    * iterations or until the desired (user-specified) error
    * is reached.
    *
    * An adaptive lambda is used; it increases when E becomes
    * better with the updated weights, and it decreases when
    * E becomes worse with the updated weights. If E becomes
    * worse, the weights are also rollbacked.
    *
    * The method prints out all the final output values as
    * well as the final total error (square root of the sum
    * of all the E^2). It also prints out the number of iterations
    * used, and the value of lambda at the end of training.
    */
   public void trainModel()
   {
      double[] errors = new double[testCases];
      double totalError = 0;

      for (int testCase = 0; testCase < testCases; testCase++)
      {
         for (int index = 0; index < nodes[0]; index++)
               activations[0][index] = inputs[testCase][index];

         testModel();
         double[] output = activations[activations.length-1];

         for (int i = 0; i < nodes[nodes.length-1]; i++)
            f[testCase][i] = output[i];

         double error = 0;

         for (int i = 0; i < nodes[nodes.length-1]; i++)
            error += (t[testCase][i]-f[testCase][i])*(t[testCase][i]-f[testCase][i]);

         errors[testCase] = error/2;

         totalError += errors[testCase]*errors[testCase];
      }//for (int testCase = 0; testCase < testCases; testCase++)

      totalError = Math.sqrt(totalError);

      int iteration = 0;

      while(!isDone(iteration, totalError))
      {
         totalError = 0;

         for (int testCase = 0; testCase < testCases; testCase++)
         {
            double[][][] originalWeights = new double[weights.length][weights[0].length][weights[0][0].length];
            for (int m = 0; m < originalWeights.length; m++)
               for (int from = 0; from < originalWeights[0].length; from++)
                  for (int to = 0; to < originalWeights[0][0].length; to++)
                     originalWeights[m][from][to] = weights[m][from][to];

            for (int index = 0; index < nodes[0]; index++)
               activations[0][index] = inputs[testCase][index]; //putting the next set of inputs
                                                                //into the activation array
            testModel();
            double[] output = activations[activations.length-1];

            for (int i = 0; i < nodes[nodes.length-1]; i++)
               f[testCase][i] = output[i];

            double error = 0;
            for (int i = 0; i < nodes[nodes.length-1]; i++)
               error += (t[testCase][i]-f[testCase][i])*(t[testCase][i]-f[testCase][i]);
            double prevError = error/2;

            for (int k = 0; k < nodes[0]; k++)
            {
               for (int j = 0; j < nodes[1]; j++)
               {
                  double aSummation = 0;

                  for (int from = 0; from < nodes[0]; from++)
                     aSummation += activations[0][from] * weights[0][from][j];

                  double aActivatedPrime = activationPrime(aSummation);

                  double iSummation = 0;

                  for (int i = 0; i < nodes[2]; i++)
                  {
                     double hSummation = 0;

                     for (int from = 0; from < nodes[1]; from++)
                        hSummation += activations[1][from] * weights[1][from][i];

                     double hActivatedPrime = activationPrime(hSummation);

                     iSummation += (t[testCase][i] - f[testCase][i])*hActivatedPrime*weights[1][j][i];
                  }//for (int i = 0; i < nodes[2]; i++)

                  weights[0][k][j] += lambda*(activations[0][k])*aActivatedPrime*iSummation;
               }//for (int j = 0; j < nodes[1]; j++)
            }//for (int k = 0; k < nodes[0]; k++)

            for (int j = 0; j < nodes[1]; j++)
            {
               for (int i = 0; i < nodes[2]; i++)
               {
                  double hSummation = 0;

                  for (int from = 0; from < nodes[1]; from++)
                     hSummation += activations[1][from] * weights[1][from][i];

                  double hActivatedPrime = activationPrime(hSummation);

                  weights[1][j][i] += lambda*(t[testCase][i] - f[testCase][i])*activations[1][j]*hActivatedPrime;
               }//for (int i = 0; i < nodes[2]; i++)
            }//for (int j = 0; j < nodes[1]; j++)

            testModel();
            output = activations[activations.length-1];

            for (int i = 0; i < nodes[nodes.length-1]; i++)
               f[testCase][i] = output[i];

            error = 0;
            for (int i = 0; i < nodes[nodes.length-1]; i++)
               error += (t[testCase][i]-f[testCase][i])*(t[testCase][i]-f[testCase][i]);
            error /= 2;

            if (error < prevError) //adapting lambda if error becomes better
            {
               if (lambda*lambdaChange < lambdaMax && lambda*lambdaChange > lambdaMin)
                  lambda *= lambdaChange;
            }//if (error < prevError)

            else                   //adapting lambda if error becomes worse
            {
               if (lambda/lambdaChange < lambdaMax && lambda/lambdaChange > lambdaMin)
                  lambda /= lambdaChange;
               weights = originalWeights;
            }//else

            errors[testCase] = error;
            totalError += errors[testCase]*errors[testCase];
         }//for (int testCase = 0; testCase < testCases; testCase++)

         totalError = Math.sqrt(totalError);

         iteration++;
      }//while(!isDone(iteration, totalError))

      totalError = 0;

      for (int testCase = 0; testCase < testCases; testCase++)
      {
         System.out.println("\n" + "test case " + testCase + " output: ");

         for (int i = 0; i < nodes[nodes.length-1]; i++)
               System.out.println(f[testCase][i]);

         totalError += errors[testCase]*errors[testCase];
      }//for (int testCase = 0; testCase < testCases; testCase++)

      System.out.println("\n" + "total error: " + Math.sqrt(totalError));
      System.out.println("\n" + "iterations: " + iteration + "\n" + "lambda: " + lambda);
   }//public void trainModel()

   /*
    * Executes the network using the configuration that has already been
    * specified at runtime. This method calculates all the activations
    * and determines the output.
    */
   public void testModel()
   {
      int layers = nodes.length; //number of layers in the network

      for (int n = 1; n < layers; n++) //setting n = 1 because the loop starts with the first hidden layer
      {
         for (int to = 0; to < nodes[n]; to++)
         {
            double activation = 0; //defaults to 0 if no weights/inputs connected to the node

            for (int from = 0; from < nodes[n-1]; from++)
               activation += weights[n-1][from][to]*activations[n-1][from];

            activations[n][to] = activation(activation);
         }//for (int to = 0; to < nodes[n]; to++)
      }//for (int n = 1; n < layers; n++)
   }//public void testModel()

   /*
    * Oversees the running of the network.
    *
    * @param args             arguments from the command line
    */
   public static void main(String[] args)
   {
      NeuralNet net = new NeuralNet();
   }//public static void main(String[] args)
}//public class NeuralNet

