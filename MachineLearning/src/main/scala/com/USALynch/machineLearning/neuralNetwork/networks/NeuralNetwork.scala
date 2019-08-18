package com.USALynch.machineLearning.neuralNetwork.networks

import java.util.concurrent.locks.ReentrantReadWriteLock

import akka.stream.Materializer
import com.USALynch.machineLearning.neuralNetwork.Exceptions.CantLockNetwork
import com.USALynch.machineLearning.neuralNetwork.activationFunction.ActivationFunction
import com.USALynch.machineLearning.neuralNetwork.costFunction.CostFunction
import com.USALynch.machineLearning.neuralNetwork.models.{Classification, Network, NetworkLayer}

import scala.concurrent.duration.Duration
import scala.concurrent.{Await, ExecutionContext, Future}

/**
  * A NeuralNetwork is an abstract class for defining how to classify and input
  * along with updating the network.
  *
  * You will implement how you wish to persist a network, and update a network.
  *
  * @author Tyler T. Lynch
  */
abstract class NeuralNetwork(implicit val ec: ExecutionContext, mat: Materializer) {

  /**
    * An activation function for the nodes.
    */
  val activationFunction: ActivationFunction

  /**
    * The cost/error of a classified input
    */
  def costFunction(
    results: Seq[Double] = Seq.empty,
    desiredOutcome: Seq[Double] = Seq.empty
  ): CostFunction

  /**
    * The network to be updated.
    *
    * Why is a def? It allows you to use asynchronous storage
    */
  def getNetwork: Future[Network]

  /**
    * To be able to save the network model.
    * This is being called once `trainBatch` is called in the `ConvolutionalNeuralNetwork`
    */
  def saveModel(network: Network): Future[Unit] = Future.successful(())


  /**
    * The actual computation on an input through a network
    * @param input the input to classify
    * @return a classified input
    */
  def feedForward(input: Seq[Double]): Future[Classification]


  /**
    * A Function to update the network weights after one iteration of training (batch or single instance)
    *
    * You can implement back propagation through the network to update the weights with this.
    * You can have a genetic algorithm do something to determine your weights.
    * You can pick random numbers
    * Basically, just have fun.
    *
    * @param classification the classification and the state of the network at that given time.
    * @param learningRate the rate at which to learn/update weights
    * @param holdBiasConstant whether to adjust the bias or not
    */
  def updateNetworkWeights(
    classification: Classification,
    learningRate: Double = 1,
    holdBiasConstant: Boolean = true): Seq[NetworkLayer]

  /**
    * @return the expected input size, to be able to verify that the input to classify is valid
    */
  def expectedInputSize: Int

  /**
    * @return the expected output size, to be able to verify that the output to train on is valid
    */
  def expectedOutputSize: Int

  /**
    * A lock on the class to be able to synchronize the network so that you can make updates to the network
    * and continue to use it at the same time. Ideally you will have an asynchronous storage of the network,
    * and you will be able to use a lock that it provides in order to update the network.
    */
  val lock: ReentrantReadWriteLock = synchronized (
    new ReentrantReadWriteLock()
  )

  /**
    * From the convolutional neural network, all processing requires obtaining a lock on the data. If you only require that you
    * can read the data, then this is all that you need. Ideally you will re-implement with using an external data source
    * that has a proper read lock.
    *
    * So long as someone does not have a lock on the write of the data, you will be able to access the data.
    *
    * @param func something that requires a lock
    * @param dur a duration to hold the lock for
    *
    * @return the result of the func
    */
  def withReadOnlyLock[T](func: => Future[T], dur: Duration = Duration.Inf)(implicit ec: ExecutionContext): T = {
    val l = lock.readLock()

    if (l.tryLock()) {
      try {
        Await.result(func, dur)
      } finally {
        l.unlock()
      }
    } else throw new CantLockNetwork
  }

  /**
    * From the convolutional neural network, all processing requires obtaining a lock on the data.
    * The reason for this is so that all processing and updating occurs on the same instance.
    *
    * This becomes an issue when you are continuously updating the network while using it. This also prevents the case in
    * which you try to compute the result of an input and then update the network. If an update on the network occurs between
    * you classifying an input and then updating it yourself, you will be updating the wrong values
    *
    * @param func something that requires a lock
    * @param dur a duration to hold the lock for
    *
    * @return the result of the func
    */
  def withLock[T](func: => Future[T], dur: Duration = Duration.Inf)(implicit ec: ExecutionContext): T = {
    val writeLock = lock.writeLock()

    if (writeLock.tryLock()) {
      try {
        Await.result(func, dur)
      } finally {
        writeLock.unlock()
      }
    } else throw new CantLockNetwork
  }

}
