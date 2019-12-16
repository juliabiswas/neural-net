
import java.io.File;
import java.io.FileWriter;
import java.io.BufferedWriter;
import java.io.IOException;
import java.util.Scanner;
import java.lang.*;

/**
 * A multilayer perceptron network of any number of layers that can take in a single set of inputs and produce
 * an output (of one or more values) or train the network on a set of input sets/true values (test cases) and
 * minimize the error by updating the weights using backpropogation.
 *
 * The class contains the following:
 *    a constructor that takes in no parameters but handles all the user-specification retrieval;
 *    a main method which oversees the program;
 *    a function that computes the value of the activation function (sigmoid) given a value;
 *    a function that computes the value of the derivative of the activation function (sigmoid) given a value;
 *    a function that determines whether the network is done training or not;
 *    a function that trains the network using back propogation; and
 *    a function that runs the network.
 *
 * The network is fully configurable by the user at runtime.
 *
 * @author Julia Biswas
 * @version December 11, 2019
 */

public class NeuralNet
{
   private int[] nodes;
   private int layers;
   private int maxNodes;
   private int testCases;

   private double[][] inputs;
   private double[][][] weights;
   private double[][] activations;
   private double[][] t; //true (expected) values
   private double[][] f; //the calculated output values
   private double[][] theta;
   private double[][] psi;
   private double[][] omega;

   private double iterations;
   private double lambda;
   private double lambdaChange;
   private double lambdaMin;
   private double lambdaMax;
   private double minRandomWeight;
   private double maxRandomWeight;
   private double desiredError;
   private boolean adaptive;

   private String inputPels; //file where the pels for training come from
   private String outputPels; //file where the pels are being outputted to after training

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
         layers = nodes.length;
         nodes[0] = inputNodes;

         for (int n = 1; n <= hiddenLayers; n++)
            nodes[n] = scanFile.nextInt();

         nodes[layers - 1] = scanFile.nextInt();

         //computing max number of nodes
         int max = 0; //index
         for (int n = 1; n < layers; n++)
            if (nodes[n] > nodes[max])
               max = n;

         maxNodes = nodes[max];

         weights = new double[layers - 1][maxNodes][maxNodes];

         //training or testing
         if (trainOrTest.equals("train"))
         {
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

               for (int m = 0; m < layers - 1; m++)
               {
                  int maxFrom = nodes[m];

                  if (m > 0)
                     maxFrom = nodes[m - 1]; //the "from" nodes are input nodes for first layer of weights

                  for (int from = 0; from < maxFrom; from++)
                  {
                     int maxTo = nodes[layers - 1];

                     if (m < layers - 2)
                        maxTo = nodes[m + 1]; //the "to" nodes are output nodes for last layer of weights

                     for (int to = 0; to < maxTo; to++)
                        weights[m][from][to] = (Math.random() * (maxRandomWeight - minRandomWeight)) + minRandomWeight;
                  }//for (int from = 0; from < maxFrom; from++)
               }//for (int m = 0; m < layers-1; m++)
            }//if (randOrSpec.equals("random"))

            else if (randOrSpec.equals("specified"))
            {
               for (int m = 0; m < layers - 1; m++)
               {
                  int maxFrom = nodes[m];

                  if (m > 0)
                     maxFrom = nodes[m - 1]; //the "from" nodes are input nodes for first layer of weights

                  for (int from = 0; from < maxFrom; from++)
                  {
                     int maxTo = nodes[layers - 1];

                     if (m < layers - 2)
                        maxTo = nodes[m + 1]; //the "to" nodes are output nodes for last layer of weights

                     for (int to = 0; to < maxTo; to++)
                        weights[m][from][to] = scanFile.nextDouble();
                  }//for (int from = 0; from < maxFrom; from++)
               }//for (int m = 0; m < layers-1; m++)
            }//else if (randOrSpec.equals("specified"))

            String adaptiveOrNot = scanFile.next();
            if (adaptiveOrNot.equals("adaptive"))
               adaptive = true;
            else
               adaptive = false;

            testCases = scanFile.nextInt();
            inputs = new double[testCases][nodes[0]];
            t = new double[testCases][nodes[layers - 1]];
            f = new double[testCases][nodes[layers - 1]];

            theta = new double[layers][maxNodes];
            psi = new double[layers][maxNodes];
            omega = new double[layers][maxNodes];

            String activationsLoc = scanFile.next();

            if (activationsLoc.equals("file") || activationsLoc.equals("pelsFile"))
            {
               inputPels = scanFile.next();

               if (activationsLoc.equals("pelsFile"))
                  outputPels = scanFile.next();

               scanFile = new Scanner(new File(inputPels));
            }//if (activationsLoc.equals("file") || activationsLoc.equals("pelsFile"))

            for (int testCase = 0; testCase < testCases; testCase++)
            {
               for (int j = 0; j < inputs[testCase].length; j++)
                  inputs[testCase][j] = scanFile.nextDouble();

               if (activationsLoc.equals("pelsFile"))
                  scanFile = new Scanner(new File(inputPels));

               for (int i = 0; i < t[testCase].length; i++)
                  t[testCase][i] = scanFile.nextDouble();

            }//for (int testCase = 0; testCase < testCases; testCase++)

            activations = new double[layers][maxNodes];

            for (int index = 0; index < nodes[0]; index++)
               activations[0][index] = inputs[0][index]; //putting the first input activations
                                                         //into the activations array

            long startTime = System.currentTimeMillis();

            trainModel();

            long endTime = System.currentTimeMillis();
            long totalTime = (endTime - startTime);
            System.out.println("time taken for training: " + totalTime + " ms");

            if (outputPels != null)
            {
               try
               {
                  BufferedWriter pelsWriter = new BufferedWriter(new FileWriter(outputPels));
                  for (double[] tc : f)
                     for (double i : tc)
                        pelsWriter.write(i + " ");

                  pelsWriter.close();
               }//try

               catch (IOException e)
               {
                  System.out.println("There was a problem writing the output out to a file.");
               }//catch (IOException e)
            }//if (outputPels != null)

         }//if (trainOrTest.equals("train"))

         else if (trainOrTest.equals("test"))
         {
            inputs = new double[1][nodes[0]];
            t = new double[1][nodes[layers - 1]];
            f = new double[1][nodes[layers - 1]];

            for (int m = 0; m < layers - 1; m++)
            {
               int maxFrom = nodes[m];

               if (m > 0)
                  maxFrom = nodes[m - 1]; //the "from" nodes are input nodes for first layer of weights

               for (int from = 0; from < maxFrom; from++)
               {
                  int maxTo = nodes[layers - 1];

                  if (m < layers - 2)
                     maxTo = nodes[m + 1]; //the "to" nodes are output nodes for last layer of weights

                  for (int to = 0; to < maxTo; to++)
                     weights[m][from][to] = scanFile.nextDouble();
               }//for (int from = 0; from < maxFrom; from++)
            }//for (int m = 0; m < layers - 1; m++)

            String fileOrInfile = scanFile.next();

            if (fileOrInfile.equals("file"))
               scanFile = new Scanner(new File(scanFile.next()));

            for (int j = 0; j < nodes[0]; j++)
               inputs[0][j] = scanFile.nextDouble();

            for (int i = 0; i < nodes[layers - 1]; i++)
               t[0][i] = scanFile.nextDouble();

            activations = new double[layers][maxNodes];

            for (int index = 0; index < nodes[0]; index++)
               activations[0][index] = inputs[0][index]; //putting the first input activations
                                                         //into the activations array
            testModel();
            double[] output = activations[activations.length - 1];

            System.out.println("\n" + "output: ");

            for (int i = 0; i < nodes[layers - 1]; i++)
               System.out.println(output[i]);
         }//else if (trainOrTest.equals("test"))
      }//try

      catch (Exception e)
      {
         System.out.println("Exception occurred trying to read " + path);
         e.printStackTrace();
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
      return 1.0 / (1.0 + Math.exp(-value));
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
      double activation = activation(value);
      return activation * (1.0 - activation);
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
      return iteration >= iterations || error <= desiredError;
   }//public boolean isDone(int iteration, double error)

   /*
    * Updates weights with the backpropagation algorithm for a
    * user-specified number of iterations or until the desired
    * (user-specified) error is reached.
    *
    * If the user desires, an adaptive lambda and weights rollback
    * are used; lambda increases when E becomes better with the
    * updated weights, and it decreases when E becomes worse with
    * the updated weights. If E becomes worse, the weights are
    * also rollbacked.
    *
    * The method prints out all the final output values as
    * well as the final total error (square root of the sum
    * of all the E^2). It also prints out the number of iterations
    * used and the value of lambda at the end of training. The time
    * spent training is also printed out, but it's done after
    * the method is done running (the code for this is in the
    * constructor).
    */
   public void trainModel()
   {
      double[] errors = new double[testCases];
      double totalError = 0.0;

      for (int testCase = 0; testCase < testCases; testCase++)
      {
         for (int index = 0; index < nodes[0]; index++)
            activations[0][index] = inputs[testCase][index];

         testModel();
         double[] output = activations[activations.length - 1];

         for (int i = 0; i < nodes[nodes.length - 1]; i++)
            f[testCase][i] = output[i];

         double error = 0.0;

         for (int i = 0; i < nodes[nodes.length - 1]; i++)
            error += (t[testCase][i] - f[testCase][i]) * (t[testCase][i] - f[testCase][i]);

         errors[testCase] = error / 2.0;

         totalError += errors[testCase] * errors[testCase];
      }//for (int testCase = 0; testCase < testCases; testCase++)

      totalError = Math.sqrt(totalError);

      int iteration = 0;

      while (!isDone(iteration, totalError))
      {
         totalError = 0.0;

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
            double[] output = activations[activations.length - 1];

            for (int i = 0; i < nodes[nodes.length - 1]; i++)
               f[testCase][i] = output[i];

            double error = 0;
            for (int i = 0; i < nodes[nodes.length - 1]; i++)
            {
               omega[layers-1][i] = t[testCase][i] - f[testCase][i];
               error += (omega[layers-1][i]) * (omega[layers-1][i]);
            }
            double prevError = error / 2.0;

            for (int j = 0; j < nodes[1]; j++) //for output weights layer
            {
               omega[1][j] = 0.0;

               for (int i = 0; i < nodes[layers-1]; i++)
               {
                  psi[layers - 1][i] = omega[layers - 1][i] * activationPrime(theta[layers - 1][i]);
                  omega[1][j] += psi[layers - 1][i] * weights[1][j][i];
                  weights[1][j][i] += lambda * activations[1][j] * psi[layers - 1][i];
               }
            }//for (int j = 0; j < nodes[1]; j++)

            for (int k = 0; k < nodes[0]; k++)
            {
               for (int j = 0; j < nodes[1]; j++)
               {
                  psi[1][j] = omega[1][j] * activationPrime(theta[1][j]);
                  weights[0][k][j] += lambda * activations[0][k] * psi[1][j];
               }
            }//for (int k = 0; k < nodes[0]; k++)

            testModel();
            output = activations[activations.length - 1];

            for (int i = 0; i < nodes[nodes.length - 1]; i++)
               f[testCase][i] = output[i];

            error = 0.0;
            for (int i = 0; i < nodes[nodes.length - 1]; i++)
            {
               omega[layers - 1][i] = t[testCase][i] - f[testCase][i];
               error += (omega[layers - 1][i]) * (omega[layers - 1][i]);
            }
            error /= 2.0;

            if (adaptive)
            {
               if (error < prevError) //adapting lambda if error becomes better
               {
                  if (lambda * lambdaChange < lambdaMax && lambda * lambdaChange > lambdaMin)
                     lambda *= lambdaChange;
               }//if (error < prevError)

               else                   //adapting lambda if error becomes worse
               {
                  if (lambda / lambdaChange < lambdaMax && lambda / lambdaChange > lambdaMin)
                     lambda /= lambdaChange;
                  weights = originalWeights;
               }//else
            }//if (adaptive)

            errors[testCase] = error;
            totalError += errors[testCase] * errors[testCase];
         }//for (int testCase = 0; testCase < testCases; testCase++)

         totalError = Math.sqrt(totalError);

         iteration++;
      }//while(!isDone(iteration, totalError))

      totalError = 0.0;

      for (int testCase = 0; testCase < testCases; testCase++)
      {
         System.out.println("\n" + "test case " + testCase + " output: ");

         for (int i = 0; i < nodes[nodes.length - 1]; i++)
            System.out.println(f[testCase][i]);

         totalError += errors[testCase] * errors[testCase];
      }//for (int testCase = 0; testCase < testCases; testCase++)

      System.out.println("\n" + "total error: " + Math.sqrt(totalError));
      System.out.println("\n" + "iterations: " + iteration + "\n" + "lambda: " + lambda);
   }//public void trainModel()

   /*
    * Executes the network using the user-specified configuration.
    * This method calculates all the activations and saves them and
    * the raw activations in instance arrays.
    */
   public void testModel()
   {
      for (int i = 0; i < nodes[layers - 1]; i++)
      {
         theta[layers-1][i] = 0.0;

         for (int j = 0; j < nodes[2]; j++)
         {
            theta[2][j] = 0.0;

            for (int k = 0; k < nodes[1]; k++)
            {
               theta[1][k] = 0.0;

               for (int m = 0; m < nodes[0]; m++)
               {
                  theta[1][k] += activations[0][m] * weights[0][m][k];
               }

               activations[1][k] = activation(theta[1][k]);
               theta[1][j] += activations[1][k] * weights[1][k][j];
            }//for (int k = 0; k < nodes[1]; k++)

            activations[2][j] = activation(theta[2][j]);
            theta[layers-1][i] += activations[2][j] * weights[2][j][i];
         }//for (int j = 0; j < nodes[2]; j++)

         activations[layers - 1][i] = activation(theta[layers-1][i]);
      }//for (int i = 0; i < nodes[layers-1]; i++)
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
