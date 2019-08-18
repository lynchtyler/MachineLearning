# Neural Networks 

## Motivation 
The motivation behind this neural network package is to make creating, testing and using an efficient
neural network in your code base easy and effortless. 

Often times neural networks are only created with research in mind, never for deployment. 
The goal with this framework is to be able to extended upon and modified to make ready for deployment
to your environment. Some examples of reasons to implement a custom NeuralNetwork or overriding
the Convolutional Neural Network include:
 - Saving the state to a NoSQL database or IMDG 
 - Taking advantage of a specific GPU 
 - Taking advantage of a specific CPU 
 - Using a custom Linear Algebra Library
 - Making some optimization to code that was overlooked 
 - Fixing a bug in the code that was not caught or that your implementation is suffering from


This framework allows you to have direct access to network and model so that 
you are able to save it (as Json) and implement into some other framework if you so wish. 
By having direct access to the network through Json, it is easier to visualize the network to
be able to make advanced alterations to it. 


### Information on Weight initialization 
https://medium.com/usf-msds/deep-learning-best-practices-1-weight-initialization-14e5c0295b94

## Pre-requisites 
- OpenCl, for LWJGL 
    - `-Dorg.lwjgl.util.Debug=true`
    - `-Dorg.lwjgl.util.DebugLoader=true` 
    - `-Dorg.lwjgl.system.allocator=system` 
     
    - Intel OpenCL SDK downloads: 
        - https://software.intel.com/en-us/opencl-sdk
        - https://software.intel.com/en-us/opencl-sdk/choose-download
        
        
## Usage 

### Basic Variable Based NeuralNetwork 
There is a basic implementation of the NeuralNetwork `BasicVariableBasedNeuralNetwork`
To use it 

```scala
import com.USALynch.machineLearning.neuralNetwork.models.{Network, KnownInput}
import com.USALynch.machineLearning.neuralNetwork.models.Network._ 
import com.USALynch.machineLearning.neuralNetwork.networks.BasicFeedForwardNetwork
import com.USALynch.machineLearning.neuralNetwork.ConvolutionalNeuralNetwork

/*
 * Create a new network with random seeds 
 */
val inputSize: Int = 5
val outputSize: Int = 3
val hiddenLayers: List[Int] = List(10,8,6)
val network: Network = Network.createNetwork(inputSize, outputSize, hiddenLayers)

/*
 * Initialize the Neural Network 
 */
val neuralNetwork = new BasicFeedForwardNetwork(network)
val cnn = ConvolutionalNeuralNetwork(neuralNetwork, 1) 

/*
 * Train the Convolutional Neural Network 
 */
val testInput: Seq[Double] = Seq(1.0d,1.0d,1.0d,1.0d,1.0d)
val testOutput: Seq[Double] = Seq(0d,1d,0d)
cnn.train(KnownInput(testInput, testOutput))

// Classify an input  
val clazz = cnn.classify(Seq(1.0d,1.0d,1.0d,1.0d,1.0d))
```
