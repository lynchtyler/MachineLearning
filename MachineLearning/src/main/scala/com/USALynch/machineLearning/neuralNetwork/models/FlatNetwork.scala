package com.USALynch.machineLearning.neuralNetwork.models

import java.util.UUID

import ai.x.play.json.Jsonx
import org.bson.types.ObjectId

/**
  * The a flat version of a single network node.
  *
  * The motivation for this is to be able to use an IMDG or external storage effectively to traverse and get node information
  *
  *
  * @param networkId a unique identifier of the network
  * @param version a unique identifier of the version, with an implicit timestamp so you know when it was last updated.
  * @param layerIndex the index of the layer in the network
  * @param nodeIndex the index of the node in the layer in the network
  * @param node the actual node information
  * @param expectedInputSize the expected input size, to be able to verify that the input to classify is valid
  * @param expectedOutputSize the expected output size, to be able to verify that the output to train on is valid
  *
  * @author Tyler T. Lynch
  */
case class FlatNetwork(
  networkId: String = UUID.randomUUID().toString,
  version: String = new ObjectId().toString,
  layerIndex: Int,
  nodeIndex: Int,
  node: NetworkNode,
  expectedInputSize: Int,
  expectedOutputSize: Int
)

object FlatNetwork {

  implicit val format = Jsonx.formatCaseClassUseDefaults[FlatNetwork]

  /**
    * Convert a network to a flattened representation
    *
    * @param network a multi dimensional network
    * @return a flattened representation of the network
    */
  def apply(network: Network): Seq[FlatNetwork] = {
    network.layers.flatMap{ layer =>
      layer.nodes.map{ node =>
        FlatNetwork(
          network.networkId,
          network.version,
          layer.index,
          node.index,
          node,
          network.expectedInputSize,
          network.expectedOutputSize
        )
      }
    }
  }

}

