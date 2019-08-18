package com.USALynch.machineLearning.neuralNetwork.networks.apache

import akka.actor.ActorSystem
import akka.stream.{ActorMaterializer, Materializer}
import com.USALynch.machineLearning.neuralNetwork.ConvolutionalNeuralNetwork
import com.USALynch.machineLearning.neuralNetwork.models._
import com.USALynch.machineLearning.neuralNetwork.networks.BasicFeedForwardNetwork
import org.scalactic.TolerantNumerics
import org.scalatest.{FlatSpec, Matchers}
import play.api.libs.json.Json

import scala.concurrent.duration.Duration
import scala.concurrent.{Await, ExecutionContext}


/*
 * The values defined as true in this spec come from:
 *    https://www.anotsorandomwalk.com/backpropagation-example-with-numbers-step-by-step/
 *
 *
 * More stuffs:
 *    https://google-developers.appspot.com/machine-learning/crash-course/backprop-scroll/
 */
class MultiOutput_Apache_FeedForwardNetworkSpec extends FlatSpec with Matchers {

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
  val learningRate = 0.01d
  val learningRate2 = 0.75d

  // The activation function needs to be a basic sigmoid/logistic function
  val cnn = ConvolutionalNeuralNetwork(neuralNetwork, 1)


  val testInput: Seq[Double] = Seq(1.0d,4.0d,5.0d)
  val testOutput: Seq[Double] = Seq(0.1d, 0.05d)

  private val clazz_1 = cnn.classify(testInput)

  it should "be able to forward propagate correctly" in {
    (clazz_1.map(_.result).getOrElse(Seq.empty[Double]) zip Array(0.8896,0.8004)).map{ case (in, out) =>
      in should be (out +- eps)
    }
  }

  private val layersWithActivations_1  = clazz_1 .get.activations

  private val lastLayer_1  = layersWithActivations_1 .last

  it should "be able to store and compute valid activation values" in {
    (lastLayer_1.nodes.head.inputs.get zip Array(0.9866,0.9950)).map{ case (in, out) =>
      in should be (out +- eps)
    }
  }

  private val hiddenLayer_1  = layersWithActivations_1 .drop(1).head
  it should "be able to update the inputs to the first node in the hidden layer" in {
    (hiddenLayer_1.nodes.head.inputs.get zip Array(1.0,4.0,5.0)).map{ case (in, out) =>
      in should be (out +- eps)
    }
  }
  it should "be able to update the inputs to the last node in the hidden layer" in {

    (hiddenLayer_1.nodes.last.inputs.get zip Array(1.0,4.0,5.0)).map{ case (in, out) =>
      in should be (out +- eps)
    }
  }






  /***************************************************************
    ******** Here is where the training gets interesting... ******
    **************************************************************/




  private val trainingResult = cnn.train(KnownInput(testInput, testOutput), learningRate)


  it should "be able calculate the training error correctly" in {
    trainingResult.avgCost should be (0.428 +- eps)
  }

  // Verify that the last layer input weights have been updated correctly
  private val newNetwork = Await.result(cnn.network.getNetwork, Duration.Inf)
  private val lastLayer = newNetwork.layers.last
  it should "be able to update the weights in the first node in the last layer" in {
    (lastLayer.nodes.head.inputWeights.map(_.weight) zip Array(0.700,0.900)).map{ case (in, out) =>
      in should be (out +- eps)
    }
  }
  it should "be able to update the weights in the last node in the last layer" in {
    (lastLayer.nodes.last.inputWeights.map(_.weight) zip Array(0.798,0.098)).map{ case (in, out) =>
      in should be (out +- eps)
    }
  }

  it should "be able to update the bias in the first node in the last layer" in {
    lastLayer.nodes.head.bias should be (0.499 +- eps)
  }
  it should "be able to update the bias in the last node in the last layer" in {
    lastLayer.nodes.last.bias should be (0.5 +- eps)
  }


  private val hiddenLayer = newNetwork.layers.drop(1).head
  it should "be able to update the weights in the first node in the hidden layer" in {
    (hiddenLayer.nodes.head.inputWeights.map(_.weight) zip Array(0.1,0.3,0.5)).map{ case (in, out) =>
      in should be (out +- eps)
    }
  }
  it should "be able to update the weights in the last node in the hidden layer" in {
    (hiddenLayer.nodes.last.inputWeights.map(_.weight) zip Array(0.2,0.4,0.6)).map{ case (in, out) =>
      in should be (out +- eps)
    }
  }


  it should "be able to update the bias in the first node in the hidden layer" in {
    hiddenLayer.nodes.head.bias should be (0.499 +- eps)
  }
  it should "be able to update the bias in the last node in the hidden layer" in {
    hiddenLayer.nodes.last.bias should be (0.499 +- eps)
  }


  private val clazz = cnn.classify(testInput)

  it should "be to reclassify with the weights and get a new value that is known to be valid out." in {
    (clazz.map(_.result).getOrElse(Seq.empty[Double]) zip Array(0.889,0.800)).map{ case (in, out) =>
      in should be (out +- eps)
    }
  }


  private val trainingResult2 = cnn.train(KnownInput(testInput, testOutput), learningRate2)
  private val processingTime = trainingResult2.processingTime
  private val timeElapsed = trainingResult2.timeElapsed
  private val clazz2 = cnn.classify(testInput)

  it should "be to reclassify with the weights and get a value closer to the desire result" in {
    val desired = testOutput.zipWithIndex.map(_.swap).toMap
    val oldOld = clazz_1.map(_.result).getOrElse(Seq.empty[Double])
    val newNew = clazz2.map(_.result).getOrElse(Seq.empty[Double])
    (newNew zip oldOld).zipWithIndex.map{ case (c, i) =>
      val (n, o) = c
      if (desired(i) < o) {
        assert(n < o)
      } else assert(n > o)
    }
  }

}
