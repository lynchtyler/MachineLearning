package com.USALynch.machineLearning.neuralNetwork.networks

import akka.actor.ActorSystem
import akka.stream.{ActorMaterializer, Materializer}
import com.USALynch.machineLearning.neuralNetwork.ConvolutionalNeuralNetwork
import com.USALynch.machineLearning.neuralNetwork.models._
import org.scalactic.TolerantNumerics
import org.scalatest.{FlatSpec, Matchers}
import play.api.libs.json.Json

import scala.concurrent.{Await, ExecutionContext}


/*
 * The values defined as true in this spec come from:
 *    https://www.anotsorandomwalk.com/backpropagation-example-with-numbers-step-by-step/
 *
 *
 * More stuffs:
 *    https://google-developers.appspot.com/machine-learning/crash-course/backprop-scroll/
 */
class TimeComparisonSpec extends FlatSpec with Matchers {

  implicit val ec: ExecutionContext = ExecutionContext.global
  implicit val system: ActorSystem = ActorSystem()
  implicit val mat: Materializer = ActorMaterializer()

  // Out epsilon for how close the test values should be
  private val eps = 1e-3
  implicit val custom = TolerantNumerics.tolerantDoubleEquality(eps)


  /*
   * Create a new network with random seeds
   */
  val inputSize: Int = 3
  val outputSize: Int = 2
  // https://www.anotsorandomwalk.com/backpropagation-example-with-numbers-step-by-step/
  val net: Network = {
    val in = s"""
      |{
      |"networkId":"89db71b2-ae3c-4380-8e66-5420ea5641bd",
      |"version":"5d2543dbd365285b5b1cd2b9",
      |"layers":[
      |   {"index":0,"nodes":[
      |     {"index":0,"bias":0,"inputWeights":[]},
      |     {"index":1,"bias":0,"inputWeights":[]},
      |     {"index":2,"bias":0,"inputWeights":[]}
      |   ]},
      |   {"index":1,"nodes":[
      |     {"index":0,"bias":0.5,"inputWeights":[
      |       {"fromIndex":0,"weight":0.10},
      |       {"fromIndex":1,"weight":0.30},
      |       {"fromIndex":2,"weight":0.50}
      |     ]},
      |     {"index":1,"bias":0.5,"inputWeights":[
      |       {"fromIndex":0,"weight":0.20},
      |       {"fromIndex":1,"weight":0.40},
      |       {"fromIndex":2,"weight":0.60}
      |     ]}
      |   ]},
      |   {"index":2,"nodes":[
      |     {"index":0,"bias":0.5,"inputWeights":[
      |       {"fromIndex":0,"weight":0.70},
      |       {"fromIndex":1,"weight":0.90}
      |     ]},
      |     {"index":1,"bias":0.5,"inputWeights":[
      |       {"fromIndex":0,"weight":0.80},
      |       {"fromIndex":1,"weight":0.10}
      |     ]}
      |   ]}
      |],
      |"expectedInputSize":${inputSize},
      |"expectedOutputSize":${outputSize}}
    """.stripMargin
    Json.parse(in).as[Network]
  }


  val network: Network = net


  /*
   * Initialize the Neural Network
   */
  val neuralNetwork = new BasicFeedForwardNetwork(network)
  val breezeNeuralNetwork = new BreezeFeedForwardNetwork(network)
  val thoughtWorksNeuralNetwork = new ThoughtWorksFeedForwardNetwork(network)
  val learningRate = 0.05d

  // The activation function needs to be a basic sigmoid/logistic function
  val cnn = ConvolutionalNeuralNetwork(neuralNetwork, 1)
  val breezeCnn = ConvolutionalNeuralNetwork(breezeNeuralNetwork, 1)
  val thoughtWorksCnn = ConvolutionalNeuralNetwork(thoughtWorksNeuralNetwork, 1)


  val testInput: Seq[Double] = Seq(1.0d,4.0d,5.0d)
  val testOutput: Seq[Double] = Seq(0.1d, 0.05d)
  val iterations = 10

  val batch = (0 until iterations).map(_ => KnownInput(testInput, testOutput))

  private val basicTrainingResult = cnn.trainBatch(batch, learningRate)
  private val breezeTrainingResult = breezeCnn.trainBatch(batch, learningRate)
  private val thoughtWorksTrainingResult = thoughtWorksCnn.trainBatch(batch, learningRate)


  it should "breeze time elapsed should be slower by roughly 2 times" in {
    println(s"${breezeTrainingResult.timeElapsed} > ${basicTrainingResult.timeElapsed}")
    assert(breezeTrainingResult.timeElapsed > basicTrainingResult.timeElapsed)
  }

  it should "breeze processing time should be slower by roughly 2 times" in {
    println(s"${breezeTrainingResult.processingTime} > ${basicTrainingResult.processingTime}")
    assert(breezeTrainingResult.processingTime > basicTrainingResult.processingTime)
  }

  it should "breeze processing time should be faster by roughly 3 times compared to thoughtWorks for just CPU" in {
    println(s"${breezeTrainingResult.processingTime} < ${thoughtWorksTrainingResult.processingTime}")
    assert(breezeTrainingResult.processingTime < thoughtWorksTrainingResult.processingTime)
  }


}
