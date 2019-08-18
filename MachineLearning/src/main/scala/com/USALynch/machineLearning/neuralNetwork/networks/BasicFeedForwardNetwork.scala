package com.USALynch.machineLearning.neuralNetwork.networks

import akka.stream.Materializer
import com.USALynch.machineLearning.neuralNetwork.activationFunction.{ActivationFunction, BasicSigmoid}
import com.USALynch.machineLearning.neuralNetwork.costFunction.{CostFunction, SumOfSquaresError}
import com.USALynch.machineLearning.neuralNetwork.models._
import org.apache.commons.math3.linear.MatrixUtils.createRealMatrix
import org.apache.commons.math3.linear.RealMatrix

import scala.concurrent.{ExecutionContext, Future}

/**
  * This implementation uses a mutable variable to maintain the state of the network and all updates on it.
  *
  * The Linear Algebra Library used for this implementation is contained within apache commons math3
  *
  * @author Tyler T. Lynch
  */
class BasicFeedForwardNetwork(
  networkSeed: Network,
  override val activationFunction: ActivationFunction = new BasicSigmoid()
)(implicit ec: ExecutionContext, mat: Materializer) extends NeuralNetwork {

  /**
    * The cost/error of a classified input
    */
  def costFunction(
    results: Seq[Double] = Seq.empty,
    desiredOutcome: Seq[Double] = Seq.empty
  ): CostFunction = new SumOfSquaresError(results, desiredOutcome)


  // A private representation of the network
  private var _network: Network = networkSeed.copy(
    activationFunction = Some(activationFunction.getClass.getCanonicalName),
    costFunction = Some(costFunction().getClass.getCanonicalName)
  )

  /**
    * The network to be updated.
    *
    * Why is a def? It allows you to use asynchronous storage
    */
  def getNetwork: Future[Network] = synchronized(Future.successful(_network))

  /**
    * @return the expected input size, to be able to verify that the input to classify is valid
    */
  def expectedInputSize: Int = synchronized(_network.expectedInputSize)

  /**
    * @return the expected output size, to be able to verify that the output to train on is valid
    */
  def expectedOutputSize: Int = synchronized(_network.expectedOutputSize)

  // TODO: It would be nice to store this in a cache
  override def saveModel(network: Network): Future[Unit] = {
    _network = network
    /*
    println(s"Current Model:\n\t${
      obj(
        "costFunction" -> costFunction().getClass.getCanonicalName,
        "activationFunction" -> activationFunction.getClass.getCanonicalName,
        "network" -> Json.toJson(network)(Network.formatNetwork)
      )
    }")
    */
    Future.successful(())
  }

  /**
    * The actual computation on an input through a network
    *
    * A good video with the notation on this:
    *   https://youtu.be/An5z8lR8asY?t=70
    *
    * TODO: It would be nice to store the activations in a cache so we don't have to worry about them.
    *
    * @param input the input to classify
    * @return a classified input
    */
  def feedForward(input: Seq[Double]): Future[Classification] = {
    for {
      network <- getNetwork
    } yield {
      val firstLayer = network.layers.headOption.getOrElse(throw new Exception(s"Why'd you create an invalid network... shmmm. no layers?"))
      val firstLayerBias = toSingleColumnMatrix(firstLayer.nodes.map(_.bias))

      // Even though the input is guaranteed to be the same size as the specified network input size, the first layer could be different.
      assert(input.length == firstLayerBias.getColumn(0).length)
      // Add the bias to each value
      val inputValues = toSingleColumnMatrix(input).add(firstLayerBias)
      val seed: FeedForwardState = FeedForwardState(inputValues, firstLayer)

      val networkWithWeights: Seq[FeedForwardState] = network.layers.drop(1).foldLeft(Seq(seed)) { case (previousLayers, layer) =>
        // The output of the previous layer is the input to this layer
        val in = previousLayers.last.output
        val columns = in.getColumn(0).length
        val rawWeights = toWeightMatrix(columns, layer.nodes)
        val biases = toSingleColumnMatrix(layer.nodes.map(_.bias))

        val sub_weightedInput = rawWeights.multiply(in)
        val weightedInput = sub_weightedInput.add(biases)

        // I'm sure there is a fancier way to do this, but I'm lazy
        val activation = toSingleColumnMatrix(weightedInput.getColumn(0).map(x => activationFunction.compute(x)))

        // Keep track of the activation of the node, to be used later.
        val layerInputs: Seq[Double] = in.getColumn(0).toSeq
        val weightedInputs: Map[Int, Double] = weightedInput.getColumn(0).toSeq.zipWithIndex.map(_.swap).toMap
        val activations: Map[Int, Double] = activation.getColumn(0).toSeq.zipWithIndex.map(_.swap).toMap

        val layerWithActivation = layer.copy(
          nodes = layer.nodes.map(node =>
            node.copy(
              inputs = Some(layerInputs),
              weightedSum = weightedInputs.get(node.index),
              activation = activations.get(node.index)
            )
          )
        )

        previousLayers ++ Seq(FeedForwardState(activation, layerWithActivation))
      }
      val output = networkWithWeights.last.output
      val tmpNetwork: Seq[NetworkLayer] = networkWithWeights.map(_.layer)

      Classification(output.getColumn(0).toSeq, tmpNetwork, Some(network.version), Some(network.networkId))
    }
  }

  case class FeedForwardState(output: RealMatrix, layer: NetworkLayer)

  private def toSingleColumnMatrix(input: Iterable[Double]): RealMatrix = {
    val emptyMatrix = createRealMatrix(input.size, 1)
    emptyMatrix.setColumn(0, input.toArray)
    emptyMatrix
  }

  private def toWeightMatrix(columns: Int, nodes: Seq[NetworkNode]): RealMatrix = {
    val emptyMatrix = createRealMatrix(nodes.length, columns)
    // Gross foreach, I know.
    nodes.zipWithIndex.foreach{ case (node, index) =>
      emptyMatrix.setRow(index, node.inputWeights.map(_.weight).toArray)
    }
    emptyMatrix
  }


  /**
    * How you wish to update the network after one iteration of training.
    *
    * Some step by step examples and documentation:
    *     https://hmkcode.github.io/ai/backpropagation-step-by-step/
    *     https://www.anotsorandomwalk.com/backpropagation-example-with-numbers-step-by-step/
    *     https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
    *     https://medium.com/@14prakash/back-propagation-is-very-simple-who-made-it-complicated-97b794c97e5c
    *
    * @param classification the classification and the state of the network at that given time.
    * @param learningRate the rate at which to learn/update weights
    */
  override def updateNetworkWeights(
    classification: Classification,
    learningRate: Double = 1,
    holdBiasConstant: Boolean = false): Seq[NetworkLayer] = {

    val network = classification.activations

    val lastLayer = network.lastOption.getOrElse(
      throw new Exception(s"Why'd you create an invalid network... shmmm. no layers?")
    )

    /*
     * dse = (d'C0/d'a_j(L)) = Sum( d'Cj ) or simply put the sum of the derivatives with respect to their desired results
     */
    val output_dse: RealMatrix = toSingleColumnMatrix(
      classification.costFunction.map(_.derivatives).getOrElse(Seq.empty[Double])
    )
    val output_adjustments = computeLayerAdjustments(lastLayer, output_dse, learningRate)
    val seed: BackwardState = adjustLayer(output_adjustments, holdBiasConstant)

    val inputLayer = network.headOption.toSeq
    val rest = network.drop(1).reverse.drop(1)
    val networkWithUpdatedWeights: Seq[NetworkLayer] = inputLayer ++ rest.foldLeft(Seq(seed)) { case (adjustedLayers, layer) =>

      val previous = adjustedLayers.last // There should for sure be a layer here
      val previousErrorPrimes: RealMatrix = previous.des

      val dse = previousErrorPrimes

      val output_adjustments = computeLayerAdjustments(layer, dse, learningRate)
      val newLayer: BackwardState = adjustLayer(output_adjustments, holdBiasConstant)

      adjustedLayers ++ Seq(newLayer)
    }.reverse.map(_.layer)


    networkWithUpdatedWeights
  }



  /**
    * Here we are making all the calculations to calculate the adjustments needed.
    *
    * dse = (d'C0/d'a_j(L)) = Sum( d'Cj ) or simply put the sum of the derivatives with respect to their desired results
    *
    * gradient_ws = Gradient of the w's d'C0/d'w(L)
    *    = (d'z(L)/d'w(L)) * (d'a(L)/d'z(L)) * (d'C0/d'a(L))
    *
    *    dws = (d'z(L)/d'w(L)) = The input a_k(L-1), i.e. the previous layer (to the left) activation values.
    *    dzs = (d'a(L)/d'z(L)) = The derivative of the z (sum of squares of the current layer), i.e. sigma'(z(L))
    *    des = (d'C0/d'a(L)) = The derivative of the sum of errors (cost'(a(L)-y))
    *        des = Sum(  (d'z_j(L)/d'a_k(L-1)) * (d'a_j(L)/d'z_j(L)) * dse  )
    *
    *           das = (d'z_j(L)/d'a_k(L-1)) = w_jk(L) or simply put the input weight from node
    *                                   (k in the previous layer, to node j in the current layer)
    *           dzs = (d'a_j(L)/d'z_j(L)) = The derivative of the z (sum of squares of the current layer), i.e. sigma'(z_j(L))
    *
    * gradient_bs = Gradient of the b's d'C0/d'b(L) or the gradient of the biases
    *    = (dzs * des)
    *
    * i.e the Gradient of the weights is the gradient of the biases * dws
    *
    * The changes to the biases and the weights are as follows
    *     changes_ws = gradient_ws scalar_multiple learning_rate
    *     changes_bs = gradient_bs scalar_multiple learning_rate
    *
    * New weights will then be:
    *     ws = ws + changes_ws
    *     bs = bs + changes_bs
    *
    *
    *
    * @param layerToAdjust the current layer that you wish to adjust
    * @param des the sum of error derivatives
    * @param learningRate the rate at which to learn/update weights
    * @return the adjustments to make
    */
  def computeLayerAdjustments(
    layerToAdjust: NetworkLayer,
    des: RealMatrix,
    learningRate: Double = 1
  ): Adjustments = {

    // dws = (d'z(L)/d'w(L)) = The input a_k(L-1), i.e. the previous layer (to the left) activation values.
    val dws: RealMatrix = toSingleColumnMatrix(
      layerToAdjust.nodes.headOption.getOrElse(throw new Exception("Malformed layer, there are no nodes."))
        .inputs.getOrElse(throw new Exception("Malformed layer, no inputs"))
    )
    // dzs = (d'a(L)/d'z(L)) = The derivative of the z (sum of squares of the current layer), i.e. sigma'(z(L))
    val dzs: RealMatrix = toSingleColumnMatrix( layerToAdjust.nodes.map(x => {
      activationFunction.derivative(x.activation.getOrElse(0d))   // weightedSum or activation
    }).toArray )

    val gradient_bs: RealMatrix = hadamardProduct(dzs, des)
    val gradient_ws: RealMatrix = dws.multiply(gradient_bs.transpose())

    val changes_bs: RealMatrix = gradient_bs.scalarMultiply(learningRate)
    val changes_ws: RealMatrix = gradient_ws.scalarMultiply(learningRate).transpose()

    Adjustments(changes_ws, changes_bs, gradient_ws, layerToAdjust)
  }


  /**
    * Hadamard Product
    * https://ml-cheatsheet.readthedocs.io/en/latest/linear_algebra.html
    */
  private def hadamardProduct(a: RealMatrix, b: RealMatrix): RealMatrix = {
    val aRows: Map[Int, Array[Double]] = a.getData.zipWithIndex.map(_.swap).toMap
    val rows: Array[Array[Double]] = b.getData.zipWithIndex.map{ case (row, index) =>
      (row zip aRows(index)).map{ case (one, two) => one*two}
    }
    createRealMatrix(rows)
  }


  /**
    * Object to hold all the adjustments to make
    *
    * @param changes_ws the adjustments to the weights
    * @param changes_bs the adjustments to the biases
    * @param des the derivatives of the error of this layer
    * @param layer the layer to adjust
    */
  case class Adjustments(
    changes_ws: RealMatrix,
    changes_bs: RealMatrix,
    des: RealMatrix,
    layer: NetworkLayer
  )


  /**
    * object to hold the adjusted layer, along with the original layer, which will
    * be used to adjust the next layer back.
    *
    * @param des the derivatives of the error in relation to the original layer
    * @param originalLayer the original layer, prior to any changes in weights/biases
    * @param layer the new layer, with changes in weights/biases
    */
  case class BackwardState(
    des: RealMatrix,
    originalLayer: NetworkLayer,
    layer: NetworkLayer
  )



  /*
   * TODO: Make Some performance optimizations here
   */
  def adjustLayer(
    adjustments: Adjustments,
    holdBiasConstant: Boolean = false
  ): BackwardState = {

    val layerToAdjust = adjustments.layer
    val biasChanges = adjustments.changes_bs
    val weightChanges = adjustments.changes_ws

    /*
     * Update the bias values
     */
    val biases = toSingleColumnMatrix(layerToAdjust.nodes.map(_.bias))
    val newBiases = biases.subtract(biasChanges)
    val biasMap: Map[Int, Double] = newBiases.getColumn(0).toSeq.zipWithIndex.map(_.swap).toMap

    /*
     * Update the weight values
     */
    val weights = createRealMatrix( layerToAdjust.nodes.map(x => {
      x.inputWeights.map(_.weight).toArray
    }).toArray )

    val newWeights = weights.subtract(weightChanges)
    val weightMap: Map[Int, Seq[InputWeight]] = newWeights.getData.zipWithIndex.map{ case (layer, index) =>
      index -> InputWeight(layer.zipWithIndex.map(_.swap).toMap)
    }.toMap

    /*
     * Assign the updates values to the weights/biases
     */
    val newLayer = layerToAdjust.copy(nodes = layerToAdjust.nodes.map{ node =>
      val newnew = weightMap(node.index)
      val biaz = if (holdBiasConstant) node.bias else biasMap(node.index)
      node.copy(inputWeights = newnew, bias = biaz)
    })
    BackwardState(adjustments.des, layerToAdjust, newLayer)
  }

}
