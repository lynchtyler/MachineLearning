package com.USALynch.machineLearning.neuralNetwork

import java.util.UUID

import akka.stream.Materializer
import akka.stream.scaladsl.{Sink, Source}
import com.USALynch.machineLearning.neuralNetwork.Exceptions.{CantLockNetwork, InvalidInputSize, InvalidOutcomeSize}
import com.USALynch.machineLearning.neuralNetwork.costFunction.CostFunction
import com.USALynch.machineLearning.neuralNetwork.models._
import com.USALynch.machineLearning.neuralNetwork.networks.NeuralNetwork
import org.joda.time.DateTime

import scala.concurrent.{ExecutionContext, Future}

/**
  * A Convolutional Neural Network abstracts the processing and training of input data from a user for a given network.
  *
  * The network implementation can vary depending on what the user wants/is using to process data.
  *
  * For example, if the user is wishing to process data on a GPU, they can use a GPU network and don't need to worry
  * about implementing it. This makes swapping out networks that are optimized for specific GPUs a breeze
  *
  * @param network the network to use
  * @param parallelism the parallelism for which to batch data
  *
  * @author Tyler T. Lynch
  */
case class ConvolutionalNeuralNetwork(
  network: NeuralNetwork,
  parallelism: Int = 8
)(implicit val ec: ExecutionContext, mat: Materializer){

  /**
    * Ensure all values are within a proper range.
    *
    * @param input values to normalize
    * @return normalized values
    */
  def normalize(input: Seq[Double]): Seq[Double] = {
    val localMin: Double = if (input.nonEmpty) input.min else 0d
    val localMax: Double = if (input.nonEmpty) input.max else 0d
    val range: Double = localMax-localMin
    input.map(x => {
      if (range > 0 ) (x-localMin)/range
      else 0d
    })

    input
  }

  /**
    * To force the user to be conscious of issues that may arise from malformed data.
    */
  @throws[InvalidInputSize]
  private def verifyInputDimensions(input: Seq[Double]): Boolean = {
    val length = input.length
    val maybe = length == network.expectedInputSize
    if (maybe) maybe else throw new InvalidInputSize(length, network.expectedInputSize)
  }

  /**
    * To force the user to be conscious of issues that may arise from malformed data.
    */
  @throws[InvalidOutcomeSize]
  private def verifyOutcomeDimensions(desiredOutcome: Seq[Double]): Boolean = {
    val length = desiredOutcome.length
    val maybe = length == network.expectedOutputSize
    if (maybe) maybe else throw new InvalidOutcomeSize(length, network.expectedOutputSize)
  }


  /**
    * To classify a batch of input data.
    *
    * Inputs are guaranteed to be processed at the same version.
    *
    * @param inputs what to classify. Tied with a set of unique identifiers so you know what results correspond to what
    *
    * @throws CantLockNetwork if you can't lock the network, that means something else is currently locking it.
    *                         You can wrap this function in some priority queue to try again later.
    * @return a map of input ids to classifications
    */
  @throws[CantLockNetwork]
  @throws[InvalidInputSize]
  def classifyBatch(inputs: Map[UUID, Seq[Double]]): Map[UUID, Classification] = {
    network.withReadOnlyLock(
      for {
        rtn <- {
          if (inputs.forall(x => verifyInputDimensions(x._2))) {
            Source(inputs).mapAsync[(UUID, Classification)](parallelism) { case (id, input) =>
              network.feedForward(normalize(input)).map(r => id -> r)
            }.runWith(Sink.seq[(UUID, Classification)]).map(_.toMap)
          } else Future.successful(Map.empty[UUID, Classification])
        }
      } yield rtn
    )
  }

  /**
    * To classify a single input
    *
    * @param input what to classify.
    *
    * @throws CantLockNetwork if you can't lock the network, that means something else is currently locking it.
    *                         You can wrap this function in some priority queue to try again later.
    * @return maybe a classifications
    */
  @throws[CantLockNetwork]
  @throws[InvalidInputSize]
  def classify(input: Seq[Double]): Option[Classification] = {
    classifyBatch(Map(UUID.randomUUID() -> input)).headOption.map(_._2)
  }

  /**
    * For a gives set of a validation data, find out what the avg costs to classify are
    *
    * @param knownInput validation data to calculate
    *
    * @throws CantLockNetwork if you can't lock the network, that means something else is currently locking it.
    *                         You can wrap this function in some priority queue to try again later.
    * @return the avg costs for the calculations on the known input data
    */
  @throws[CantLockNetwork]
  @throws[InvalidInputSize]
  @throws[InvalidOutcomeSize]
  def validateNetwork(knownInput: Seq[KnownInput]): ValidationResults = {
    network.withReadOnlyLock(
      if (knownInput.forall(x => verifyInputDimensions(x.input))) {
        for {
          classifications: Seq[(Classification, Double, Int)] <- {
            Future.sequence(knownInput.map{ validationData =>
              for {
                classified <- network.feedForward(normalize(validationData.input))
              } yield {
                val (classifiedResult, i) = classified.result.zipWithIndex.maxBy(_._1)
                val classifiedResultIndex = if (network.activationFunction.isActive(classifiedResult)) i else -1
                val expected = validationData.result.filter(x => network.activationFunction.isActive(x))
                val expectedResultIndex = if (expected.nonEmpty) validationData.result.zipWithIndex.maxBy(_._1)._2 else -1

                val success = if (expectedResultIndex == classifiedResultIndex) 1.0d else 0.0d
                val cf: CostFunction = network.costFunction(classified.result, validationData.result)
                (classified.copy(costFunction = Some(cf)), success, expectedResultIndex)
              }
            })
          }
        } yield {
          val costs = CostFunction.avgGradient( classifications.flatMap(_._1.costFunction) )
          val cost = if (costs.nonEmpty) costs.max else -1d
          val avgCost = if (costs.nonEmpty) costs.sum/costs.length.toDouble else -1d
          val correctValidations = classifications.map(_._2)
          val byCat = classifications.groupBy(_._3).map { case (index, rest) =>
            val success = if (rest.nonEmpty) {
              rest.map(_._2).sum / rest.length.toDouble
            } else -1
            index -> success
          }.toSeq.sortBy(_._1).toMap
          val recall = if (correctValidations.nonEmpty) correctValidations.sum/correctValidations.length.toDouble else -1d
          ValidationResults(cost, avgCost, recall, byCat)
        }

      } else Future.successful(ValidationResults(-1d, -1d, -1d, Map.empty))
    )
  }

  case class ValidationResults(
    cost: Double,
    avgCost: Double,
    overallRecall: Double,
    recallBreakdown: Map[Int, Double]
  ) {
    def avgCostByBreakdown = if (recallBreakdown.nonEmpty) {
      recallBreakdown.values.map(x => 1.0d-x).sum/recallBreakdown.size.toDouble
    } else 1d
  }

  /**
    * To train a batch of data and only back propagate on the full batch of data.
    *
    * This assumes that you will be mini batching:
    *   https://towardsdatascience.com/gradient-descent-algorithm-and-its-variants-10f652806a3
    *   http://www.ashukumar27.io/MIni-Batch-Gradient-Descent/
    * However, if you do not want to mini batch:
    *   - To achieve Batch, make your batch size the entire thin.
    *   - To achieve Stochastic, make your batch size one
    *
    * @param knownInput known input data
    * @param learningRate the rate at which to learn/update weights
    *
    * @throws CantLockNetwork if you can't lock the network, that means something else is currently locking it.
    *                         You can wrap this function in some priority queue to try again later. This requires
    *                         both read and write locking
    * @return the cost of the training
    */
  @throws[CantLockNetwork]
  @throws[InvalidInputSize]
  @throws[InvalidOutcomeSize]
  def trainBatch(
    knownInput: scala.collection.immutable.Iterable[KnownInput],
    learningRate: Double = 1,
    holdBiasConstant: Boolean = true,
    activationBasedCost: Boolean = true): TrainingRun = {

    val startTime = DateTime.now
    val validInput = knownInput.forall(x => verifyInputDimensions(x.input))
    val validOutput = knownInput.forall(x => verifyOutcomeDimensions(x.result))

    val classificationsF: Future[Seq[Classification]] = if (validInput && validOutput) {

      Source(knownInput).mapAsync[Classification](parallelism){ ki =>
        val input: Seq[Double] = normalize(ki.input)
        val desiredOutcome: Seq[Double] = normalize(ki.result)

        for {
          classified <- network.feedForward(normalize(input))
        } yield {
          val r = if (!activationBasedCost) classified.result else {
            classified.result.map(x => if (network.activationFunction.isActive(x)) 1d else 0d)
          }
          val cf: CostFunction = network.costFunction(r, desiredOutcome)
          classified.copy(costFunction = Some(cf))
        }
      }.runWith(Sink.seq[Classification])

    } else Future.successful(Seq.empty[Classification])

    network.withLock({
      val _startTime = DateTime.now

      for {
        startingState <- network.getNetwork
        groupedLayers: Map[Int, NetworkLayer] = startingState.layers.map(x => x.index -> x).toMap
        classifications: Seq[Classification] <- classificationsF
        costFunctions: Seq[CostFunction] = classifications.flatMap(_.costFunction)
        avgCosts = CostFunction.avgGradient(costFunctions)

        // TODO: It might be nice to be able to memoize the updates of values.
        updatedNetworks: Seq[(Int, FlatNetwork)] = classifications.flatMap { classification =>
          val layers = network.updateNetworkWeights(classification, learningRate, holdBiasConstant)
          FlatNetwork(startingState.copy(layers = layers)).map( x =>
            x.layerIndex -> x
          )
        }

        // TODO: This is gross, optimize it
        newLayers = updatedNetworks.groupBy(_._1).map{ case (layerIndex, layered) =>

          val nodeMap: Map[Int, Seq[NetworkNode]] = layered.map(_._2).map(_.node).groupBy(_.index)
          val newNodes: Seq[NetworkNode] = nodeMap.map{ case (_, nodes) =>
            val startingNode = nodes.head
            val len = nodes.length
            if (len > 1) {
              val summed = nodes.drop(1).foldLeft(startingNode) { case (accum, cur) =>
                val in = InputWeight.toMap(accum.inputWeights)
                val mid = InputWeight.toMap(cur.inputWeights)
                val weights = (in zip mid).map { case (one, two) =>
                  InputWeight(one._1, one._2 + two._2)
                }.toSeq.sortBy(_.fromIndex)
                accum.copy(inputWeights = weights)
              }
              val avgWeights = summed.inputWeights.map{x => x.copy(weight = x.weight/len)}
              summed.copy(inputWeights = avgWeights)
            } else startingNode
          }.toSeq.sortBy(_.index)
          val originalLayer: NetworkLayer = groupedLayers.getOrElse(layerIndex, throw new Exception("Somehow we got a new layer"))

          originalLayer.copy(
            nodes = newNodes
          )
        }.toSeq.sortBy(_.index)
        newNetwork = startingState.copy(layers = newLayers)

        _ <- network.getNetwork.map(n => network.saveModel(newNetwork))
      } yield {
        TrainingRun(avgCosts, startTime, _startTime)
      }
    })
  }

  /**
    * To train a single input of known data
    *
    * @param knownInput a single known instance
    * @param learningRate the rate at which to learn/update weights
    *
    * @throws CantLockNetwork if you can't lock the network, that means something else is currently locking it.
    *                         You can wrap this function in some priority queue to try again later.
    */
  @throws[CantLockNetwork]
  @throws[InvalidInputSize]
  @throws[InvalidOutcomeSize]
  def train(knownInput: KnownInput, learningRate: Double = 1): TrainingRun = trainBatch(scala.collection.immutable.Iterable(knownInput), learningRate)

}
