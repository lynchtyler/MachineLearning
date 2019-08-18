package com.USALynch.machineLearning.neuralNetwork.models

import ai.x.play.json.Jsonx
import play.api.libs.json.OFormat

/**
  * A node in a network layer
  *
  * @param index the index of the node
  * @param bias the bias of the node
  *
  * @param inputWeights the input weights of the node
  * @param activationFunction the name or descriptor of the activation function
  * @param costFunction the name or descriptor of the cost function
  *
  * @param inputs the input values to the node
  * @param weightedSum the weighted sum that is used by the activation function. This is also referred to as z.
  *                    Note that this is being used by the forward propagation to make easy work
  *                    of the back propagation
  * @param activation the activation of the node
  *
  * @author Tyler T. Lynch
  */
case class NetworkNode(
  index: Int,
  bias: Double,
  inputWeights: Seq[InputWeight] = Seq.empty[InputWeight],
  activationFunction: Option[String] = None,
  costFunction: Option[String] = None,
  inputs: Option[Seq[Double]] = None,
  weightedSum: Option[Double] = None,
  activation: Option[Double] = None
) {

  /**
    * @return a map of the node index to value (input weight or bias)
    */
  def mapBias: Map[Int, Double] = Map(index -> bias)

  /**
    * To get a map of the input weights to a node
    */
  def mapInputWeights: Map[Int, Map[Int, Double]] = Map(
    index -> InputWeight.toMap(inputWeights)
  )

}

object NetworkNode {

  implicit val format: OFormat[NetworkNode] = Jsonx.formatCaseClassUseDefaults[NetworkNode]

  /**
    * Create a mapping of indices to values from a seq of network layer
    *
    * @param nodes a seq of network nodes
    * @return a map of indices to node biases
    */
  def mapBias(nodes: Seq[NetworkNode]): Map[Int, Double] = {
    nodes.flatMap(_.mapBias).toMap
  }

  /**
    * Create a mapping of indices to input weights for a layer
    *
    * @param nodes a seq of network nodes for a given layer
    * @return a map of indices to input weights for each node
    */
  def mapInputWeights(nodes: Seq[NetworkNode]): Map[Int, Map[Int, Double]] = {
    nodes.flatMap(_.mapInputWeights).toMap
  }

  /**
    * Create a new layer from a map of indices to values (biases and weights
    *
    * @param biases the index of a node mapped to a bias
    * @param weights the index of a node mapped to the input weights
    * @param activationFunction the name or descriptor of the activation function
    * @param costFunction the name or descriptor of the cost function
    * @return a seq of network nodes
    */
  def createLayer(
    biases: Map[Int, Double],
    weights: Map[Int, Map[Int, Double]],
    activationFunction: Option[String] = None,
    costFunction: Option[String] = None): Seq[NetworkNode] = {

    (biases zip weights).map { case (_bias, _weights) =>
      val index = _bias._1
      val bias = _bias._2
      val inputWeights = InputWeight(_weights._2)
      NetworkNode(index, bias, inputWeights, activationFunction, costFunction)
    }.toSeq
  }

  /**
    * Create a new layer from a map of indices to values (biases and weights
    *
    * @param nodes to create a map from
    * @return a map of indices to nodes
    */
  def toMap(nodes: Seq[NetworkNode]): Map[Int, NetworkNode] = {
    nodes.map(x => x.index -> x).toMap
  }

  /**
    * Create a new layer from a map of of nodes. Yes, I'm lazy
    *
    * @param nodes for a layer
    * @return a seq of network nodes for the layer
    */
  def apply(nodes: Map[Int, NetworkNode]): Seq[NetworkNode] =  nodes.values.toSeq
}



