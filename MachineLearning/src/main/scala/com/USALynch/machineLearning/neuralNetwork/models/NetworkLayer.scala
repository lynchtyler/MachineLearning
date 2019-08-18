package com.USALynch.machineLearning.neuralNetwork.models

import ai.x.play.json.Jsonx
import play.api.libs.json.OFormat

/**
  * A layer in a network
  *
  * @author Tyler T. Lynch
  */
case class NetworkLayer(
  index: Int,
  nodes: Seq[NetworkNode]
) {

  /**
    * @return a map of the layer index to nodes as to assist with get information for a node
    */
  def toMap: Map[Int, Map[Int, NetworkNode]] = Map(index -> NetworkNode.toMap(nodes))
}

object NetworkLayer {

  implicit val format: OFormat[NetworkLayer] = Jsonx.formatCaseClassUseDefaults[NetworkLayer]

  /**
    * Create a mapping of indices to values from a seq of network layers
    *
    * @param layers a seq of network layers
    * @return a map of indices to network values (input weights or biases)
    */
  def toMap(layers: Seq[NetworkLayer]): Map[Int, Map[Int, NetworkNode]] = {
    layers.flatMap(_.toMap).toMap
  }

  /**
    * Create a new network from a map of indices to values
    *
    * @param layers a map of indices to network values (input weights or biases)
    * @return a seq of network layers
    */
  def apply(layers: Map[Int, Map[Int, NetworkNode]]): Seq[NetworkLayer] = {
    layers.map { case (index, nodes) =>
      NetworkLayer(index, NetworkNode(nodes))
    }.toSeq
  }
}


