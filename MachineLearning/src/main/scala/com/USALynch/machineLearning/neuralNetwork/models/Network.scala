package com.USALynch.machineLearning.neuralNetwork.models

import java.util.UUID

import ai.x.play.json.Jsonx
import org.bson.types.ObjectId
import play.api.libs.json.OFormat

import scala.util.Random

/**
  * The underlining network that holds the state of the network.
  *
  * This network can be persisted as json.
  *
  * @param networkId a unique identifier of the network
  * @param version a unique identifier of the version, with an implicit timestamp so you know when it was last updated.
  * @param layers the layers of nodes in the network
  * @param expectedInputSize the expected input size, to be able to verify that the input to classify is valid
  * @param expectedOutputSize the expected output size, to be able to verify that the output to train on is valid
  * @param activationFunction the name or descriptor of the activation function
  * @param costFunction the name or descriptor of the cost function
  *
  * @author Tyler T. Lynch
  */
case class Network(
  networkId: String = UUID.randomUUID().toString,
  version: String = new ObjectId().toString,
  layers: Seq[NetworkLayer] = Seq.empty[NetworkLayer],
  expectedInputSize: Int,
  expectedOutputSize: Int,
  activationFunction: Option[String] = None,
  costFunction: Option[String] = None
)

object Network {

  implicit val formatNetwork: OFormat[Network] = Jsonx.formatCaseClassUseDefaults[Network]

  /**
    * Creates a Network
    *
    * @param inputSize the size of the input
    * @param outputSize the size of the output
    * @param hiddenLayers the hidden layers
    * @return a new network
    */
  def createNetwork(
    inputSize: Int,
    outputSize: Int,
    hiddenLayers: List[Int] = List.empty[Int],
    randomness: Int = 1,
    withBiasWeights: Boolean = false): Network = {

    val baseNetwork = createNetworkLayers(inputSize, outputSize, hiddenLayers, withBiasWeights, randomness)
    Network(
      layers = baseNetwork,
      expectedInputSize = inputSize,
      expectedOutputSize = outputSize
    )
  }

  /**
    * Creates a seq of network layers for a network
    *
    * https://medium.com/usf-msds/deep-learning-best-practices-1-weight-initialization-14e5c0295b94
    *
    * @param inputSize the size of the input
    * @param outputSize the size of the output
    * @param hiddenLayers the hidden layers
    * @return a seq of network layers for the network
    */
  private def createNetworkLayers(
    inputSize: Int,
    outputSize: Int,
    hiddenLayers: List[Int] = List.empty[Int],
    withBiasWeights: Boolean = false,
    randomness: Int = 1): Seq[NetworkLayer] = {

    val r = Random
    def getEvenOdd = if (r.nextDouble() < 0.5) 1 else 1
    def nextDouble(nodes: Int) = {
      val equal = 1.0/nodes.toDouble
      val outputEqual = 1.0/outputSize.toDouble
      val half = 0.5
      val percent = 1.0/(nodes+1).toDouble
      val leftOver = 1.0-percent
      val seed = 0d
      r.shuffle((0 until nodes).foldLeft(Seq(seed)){ case (accum, cur) =>
        val rand = getEvenOdd*r.nextDouble()*randomness
        val lessPrevious = accum.last-percent
        val halfPrevious = accum.last/2.0
        accum ++ Seq(seed)
      }).zipWithIndex.map(_.swap).toMap
    }
    def nextBias(nodes: Int) = {
      if (withBiasWeights) nextDouble(nodes)
      else (0 until nodes).map( i => i -> 0d).toMap
    }

    val totalLayers = hiddenLayers.size+2
    val inputLayer: NetworkLayer = NetworkLayer(0, (0 until inputSize).map(i => {
      NetworkNode(i, 0d) // We don't need input weights
    }))

    val lastHiddenLayer: Int = hiddenLayers.lastOption.getOrElse(inputSize)
    val outputBiasWeights = nextBias(outputSize)
    val outputLayer: NetworkLayer = NetworkLayer(totalLayers-1, (0 until outputSize).map(li => {
      val weights = nextDouble(lastHiddenLayer)
      NetworkNode(li, outputBiasWeights(li), (0 until lastHiddenLayer).map(i => InputWeight(i, weights(i))))
    }))


    val layers: Seq[NetworkLayer] = hiddenLayers.zipWithIndex.foldLeft(Seq(inputLayer)) { case (accum, cur) =>
      val (layerSize, i) = cur
      val biasWeights = nextBias(layerSize)
      val previousSize = accum.lastOption.map(_.nodes.size).getOrElse(0)
      accum ++ Seq(NetworkLayer(i+1, (0 until layerSize).map(li => {
        val weights = nextDouble(previousSize)
        NetworkNode(li, biasWeights(li), (0 until previousSize).map(i => InputWeight(i, weights(i))))
      })))
    }

    layers ++ Seq(outputLayer)
  }


  /**
    * Transform a mapped network to a network
    *
    * @param network a mapped network
    * @return a network
    */
  def fromMapped(network: MappedNetwork): Network = {
    Network(
      network.networkId,
      network.version,
      NetworkLayer(network.layers),
      network.expectedInputSize,
      network.expectedOutputSize
    )
  }


}
