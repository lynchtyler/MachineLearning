package com.USALynch.machineLearning.neuralNetwork.models

import java.util.UUID

import org.bson.types.ObjectId

/**
  * The a mapped version of the network.
  *
  * Why a mapped version and a normal version?
  *   The mapped version is not easily persisted to external
  *   storage but it is easier to work with (getting and settings values).
  *
  * To effectively update nodes in a layer, iterate over the map, updating them individually, then copy the object
  * and reset the layer(s)
  *
  * To effectively use this in an IMDG, flatten the network to an iterable to be able to use indices of nodes as lookup keys
  *
  * @param networkId a unique identifier of the network
  * @param version a unique identifier of the version, with an implicit timestamp so you know when it was last updated.
  * @param layers the layers of nodes in the network
  * @param expectedInputSize the expected input size, to be able to verify that the input to classify is valid
  * @param expectedOutputSize the expected output size, to be able to verify that the output to train on is valid
  *
  * @author Tyler T. Lynch
  */
case class MappedNetwork(
  networkId: String = UUID.randomUUID().toString,
  version: String = new ObjectId().toString,
  layers: Map[Int, Map[Int, NetworkNode]] = Map.empty[Int, Map[Int, NetworkNode]],
  expectedInputSize: Int,
  expectedOutputSize: Int
) {

  /**
    * For a given network, get the specific node (bias and input weights)
    *
    * @param x the layer
    * @param y the depth of the node
    * @return a specific input weight
    */
  def getNode(x: Int, y: Int): Option[NetworkNode] = {
    layers.get(x).flatMap(_.get(y))
  }

  /**
    * For a given network, set a specific node
    *
    * Note to the user:
    *   It is more effective to update the layers in a batch, less traversing
    *
    * @param x the layer
    * @param y the depth of the node
    * @param node the new node
    *
    * @return an update network
    */
  def setNode(x: Int, y: Int, node: NetworkNode): MappedNetwork = {
    this.copy(
      layers = layers.map{ case (i, nodes) =>
        i -> nodes.map{ case (j, _node) =>
          if (i == x && j == y) j -> node
          else j -> _node
        }
      }
    )
  }

}

object MappedNetwork {

  /**
    * Transform a network to a mapped network
    *
    * @param network a network
    * @return a mapped network
    */
  def fromNetwork(network: Network): MappedNetwork = {
    MappedNetwork(
      network.networkId,
      network.version,
      NetworkLayer.toMap(network.layers),
      network.expectedInputSize,
      network.expectedOutputSize
    )
  }
}

