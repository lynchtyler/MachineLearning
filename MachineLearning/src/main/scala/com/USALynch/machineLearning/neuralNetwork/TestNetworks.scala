package com.USALynch.machineLearning.neuralNetwork

import akka.actor.ActorSystem
import akka.stream.{ActorMaterializer, Materializer}
import com.USALynch.machineLearning.neuralNetwork.models.{KnownInput, Network}
import com.USALynch.machineLearning.neuralNetwork.networks.BasicFeedForwardNetwork

import scala.concurrent.ExecutionContext

object TestNetworks {

  implicit val ec: ExecutionContext = ExecutionContext.global
  implicit val system: ActorSystem = ActorSystem()
  implicit val mat: Materializer = ActorMaterializer()

  def main(args: Array[String]): Unit = {

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
    val trainingResult = cnn.train(KnownInput(testInput, testOutput))
    println(trainingResult)

    // Classify an input
    val clazz = cnn.classify(Seq(1.0d,1.0d,1.0d,1.0d,1.0d))
    println(clazz)
  }
}
