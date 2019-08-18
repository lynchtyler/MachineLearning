package com.USALynch.machineLearning.neuralNetwork.models

import ai.x.play.json.Jsonx
import play.api.libs.json.OFormat

/**
  * A node in a network
  *
  * @author Tyler T. Lynch
  */
case class InputWeight(
  fromIndex: Int,
  weight: Double
) {

  /**
    * @return a map of the node index to input weight
    */
  def toMap: Map[Int, Double] = Map(fromIndex -> weight)
}


object InputWeight {

  implicit val format: OFormat[InputWeight] = Jsonx.formatCaseClassUseDefaults[InputWeight]

  /**
    * Create a mapping of indices to values from a seq of network layer
    *
    * @param weights a seq of input weights for a node
    * @return a map of indices to node values (input weights or biases)
    */
  def toMap(weights: Seq[InputWeight]): Map[Int, Double] = {
    weights.flatMap(_.toMap).toMap
  }

  /**
    * Create a new layer from a map of indices to values
    *
    * @param weights a map of indices to input weights
    * @return a seq of network nodes
    */
  def apply(weights: Map[Int, Double]): Seq[InputWeight] = {
    weights.map { case (index, weight) =>
      InputWeight(index, weight)
    }.toSeq
  }
}



