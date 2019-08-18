package com.USALynch.machineLearning.neuralNetwork.networks.thoughtWorks

import akka.actor.ActorSystem
import akka.stream.{ActorMaterializer, Materializer}
import com.USALynch.machineLearning.neuralNetwork.ConvolutionalNeuralNetwork
import com.USALynch.machineLearning.neuralNetwork.models._
import com.USALynch.machineLearning.neuralNetwork.networks.ThoughtWorksFeedForwardNetwork
import org.scalactic.TolerantNumerics
import org.scalatest.{FlatSpec, Matchers}
import play.api.libs.json.Json

import scala.concurrent.duration.Duration
import scala.concurrent.{Await, ExecutionContext}


/*
 * This is to test:
 *    - That the feed forward functions for zig-zagging and then reducing to one
 *    - That the back propagtion functions for
 *        - going from a single output and then back up to multiple levels
 *        - zig-zagging between sizes.
 *
 * These tests ensure that the linear algebra operation dimensions are correct
 */
class ZigZag_thoughtWorks_FeedForwardNetworkSpec extends FlatSpec with Matchers {

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
  val outputSize: Int = 1
  /*
   * 0    1    2    3    4    5    6    7
   * 3 -> 5 -> 3 -> 5 -> 4 -> 3 -> 2 -> 1
   */
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
      |
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
      |     ]},
      |     {"index":2,"bias":0.5,"inputWeights":[
      |       {"fromIndex":0,"weight":0.20},
      |       {"fromIndex":1,"weight":0.40},
      |       {"fromIndex":2,"weight":0.60}
      |     ]},
      |     {"index":3,"bias":0.5,"inputWeights":[
      |       {"fromIndex":0,"weight":0.20},
      |       {"fromIndex":1,"weight":0.40},
      |       {"fromIndex":2,"weight":0.60}
      |     ]},
      |     {"index":4,"bias":0.5,"inputWeights":[
      |       {"fromIndex":0,"weight":0.20},
      |       {"fromIndex":1,"weight":0.40},
      |       {"fromIndex":2,"weight":0.60}
      |     ]}
      |   ]},
      |
      |   {"index":2,"nodes":[
      |     {"index":0,"bias":0.5,"inputWeights":[
      |       {"fromIndex":0,"weight":0.10},
      |       {"fromIndex":1,"weight":0.30},
      |       {"fromIndex":2,"weight":0.50},
      |       {"fromIndex":3,"weight":0.50},
      |       {"fromIndex":4,"weight":0.50}
      |     ]},
      |     {"index":1,"bias":0.5,"inputWeights":[
      |       {"fromIndex":0,"weight":0.10},
      |       {"fromIndex":1,"weight":0.30},
      |       {"fromIndex":2,"weight":0.50},
      |       {"fromIndex":3,"weight":0.50},
      |       {"fromIndex":4,"weight":0.50}
      |     ]},
      |     {"index":2,"bias":0.5,"inputWeights":[
      |       {"fromIndex":0,"weight":0.10},
      |       {"fromIndex":1,"weight":0.30},
      |       {"fromIndex":2,"weight":0.50},
      |       {"fromIndex":3,"weight":0.50},
      |       {"fromIndex":4,"weight":0.50}
      |     ]}
      |   ]},
      |
      |   {"index":3,"nodes":[
      |     {"index":0,"bias":0.5,"inputWeights":[
      |       {"fromIndex":0,"weight":0.10},
      |       {"fromIndex":1,"weight":0.30},
      |       {"fromIndex":2,"weight":0.50}
      |     ]},
      |     {"index":1,"bias":0.5,"inputWeights":[
      |       {"fromIndex":0,"weight":0.20},
      |       {"fromIndex":1,"weight":0.40},
      |       {"fromIndex":2,"weight":0.60}
      |     ]},
      |     {"index":2,"bias":0.5,"inputWeights":[
      |       {"fromIndex":0,"weight":0.20},
      |       {"fromIndex":1,"weight":0.40},
      |       {"fromIndex":2,"weight":0.60}
      |     ]},
      |     {"index":3,"bias":0.5,"inputWeights":[
      |       {"fromIndex":0,"weight":0.20},
      |       {"fromIndex":1,"weight":0.40},
      |       {"fromIndex":2,"weight":0.60}
      |     ]},
      |     {"index":4,"bias":0.5,"inputWeights":[
      |       {"fromIndex":0,"weight":0.20},
      |       {"fromIndex":1,"weight":0.40},
      |       {"fromIndex":2,"weight":0.60}
      |     ]}
      |   ]},
      |
      |
      |   {"index":4,"nodes":[
      |     {"index":0,"bias":0.5,"inputWeights":[
      |       {"fromIndex":0,"weight":0.10},
      |       {"fromIndex":1,"weight":0.30},
      |       {"fromIndex":2,"weight":0.50},
      |       {"fromIndex":3,"weight":0.50},
      |       {"fromIndex":4,"weight":0.50}
      |     ]},
      |     {"index":1,"bias":0.5,"inputWeights":[
      |       {"fromIndex":0,"weight":0.10},
      |       {"fromIndex":1,"weight":0.30},
      |       {"fromIndex":2,"weight":0.50},
      |       {"fromIndex":3,"weight":0.50},
      |       {"fromIndex":4,"weight":0.50}
      |     ]},
      |     {"index":2,"bias":0.5,"inputWeights":[
      |       {"fromIndex":0,"weight":0.10},
      |       {"fromIndex":1,"weight":0.30},
      |       {"fromIndex":2,"weight":0.50},
      |       {"fromIndex":3,"weight":0.50},
      |       {"fromIndex":4,"weight":0.50}
      |     ]},
      |     {"index":3,"bias":0.5,"inputWeights":[
      |       {"fromIndex":0,"weight":0.10},
      |       {"fromIndex":1,"weight":0.30},
      |       {"fromIndex":2,"weight":0.50},
      |       {"fromIndex":3,"weight":0.50},
      |       {"fromIndex":4,"weight":0.50}
      |     ]}
      |   ]},
      |
      |
      |   {"index":5,"nodes":[
      |     {"index":0,"bias":0.5,"inputWeights":[
      |       {"fromIndex":0,"weight":0.10},
      |       {"fromIndex":1,"weight":0.30},
      |       {"fromIndex":2,"weight":0.50},
      |       {"fromIndex":3,"weight":0.50}
      |     ]},
      |     {"index":1,"bias":0.5,"inputWeights":[
      |       {"fromIndex":0,"weight":0.10},
      |       {"fromIndex":1,"weight":0.30},
      |       {"fromIndex":2,"weight":0.50},
      |       {"fromIndex":3,"weight":0.50}
      |     ]},
      |     {"index":2,"bias":0.5,"inputWeights":[
      |       {"fromIndex":0,"weight":0.10},
      |       {"fromIndex":1,"weight":0.30},
      |       {"fromIndex":2,"weight":0.50},
      |       {"fromIndex":3,"weight":0.50}
      |     ]}
      |   ]},
      |
      |   {"index":6,"nodes":[
      |     {"index":0,"bias":0.5,"inputWeights":[
      |       {"fromIndex":0,"weight":0.70},
      |       {"fromIndex":1,"weight":0.90},
      |       {"fromIndex":1,"weight":0.50}
      |     ]},
      |     {"index":1,"bias":0.5,"inputWeights":[
      |       {"fromIndex":0,"weight":0.80},
      |       {"fromIndex":1,"weight":0.10},
      |       {"fromIndex":1,"weight":0.50}
      |     ]}
      |   ]},
      |
      |
      |   {"index":7,"nodes":[
      |     {"index":0,"bias":0.5,"inputWeights":[
      |       {"fromIndex":0,"weight":0.10},
      |       {"fromIndex":1,"weight":0.90}
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
  val neuralNetwork = new ThoughtWorksFeedForwardNetwork(network)
  val learningRate = 0.01d

  // The activation function needs to be a basic sigmoid/logistic function
  val cnn = ConvolutionalNeuralNetwork(neuralNetwork, 1)


  val testInput: Seq[Double] = Seq(1.0d,4.0d,5.0d)
  val testOutput: Seq[Double] = Seq(0.21d)

  private val clazz_1 = cnn.classify(testInput)

  it should "be able to forward propagate correctly" in {
    (clazz_1.map(_.result).getOrElse(Seq.empty[Double]) zip Array(0.794)).map{ case (in, out) =>
      in should be (out +- eps)
    }
  }

  private val layersWithActivations_1  = clazz_1 .get.activations



  /***************************************************************
    ******** Here is where the training gets interesting... ******
    **************************************************************/




  private val trainingResult = cnn.train(KnownInput(testInput, testOutput), learningRate)


  it should "be able calculate the training error correctly" in {
    trainingResult.avgCost should be (0.312 +- eps)
  }

  // Verify that the last layer input weights have been updated correctly
  private val newNetwork = Await.result(cnn.network.getNetwork, Duration.Inf)
  private val lastLayer = newNetwork.layers.last
  it should "be able to update the weights in the first node in the last layer" in {
    (lastLayer.nodes.head.inputWeights.map(_.weight) zip Array(0.099,0.899)).map{ case (in, out) =>
      in should be (out +- eps)
    }
  }
  it should "be able to update the weights in the last node in the last layer" in {
    (lastLayer.nodes.last.inputWeights.map(_.weight) zip Array(0.099,0.899)).map{ case (in, out) =>
      in should be (out +- eps)
    }
  }


  private val clazz = cnn.classify(testInput)

  it should "be to reclassify with the weights and get a new value that is known to be valid out." in {
    (clazz.map(_.result).getOrElse(Seq.empty[Double]) zip Array(0.793)).map{ case (in, out) =>
      in should be (out +- eps)
    }
  }


}
