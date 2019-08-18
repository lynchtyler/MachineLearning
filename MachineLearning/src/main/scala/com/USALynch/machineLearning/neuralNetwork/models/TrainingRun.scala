package com.USALynch.machineLearning.neuralNetwork.models

import org.joda.time.DateTime

/**
  * The result of training a batch or single instance of known data
  *
  * @param costs for the training
  * @param startTime the start time of the processing
  * @param startTimeWithLock the start time of the processing once the lock is obtained
  * @param endTime the endTime of the processing
  *
  * @author Tyler T. Lynch
  */
case class TrainingRun(
  costs: Seq[Double] = Seq.empty[Double],
  startTime: DateTime,
  startTimeWithLock: DateTime,
  endTime: DateTime = DateTime.now
) {

  def timeElapsed: Long = endTime.getMillis-startTime.getMillis

  def processingTime: Long = endTime.getMillis-startTimeWithLock.getMillis

  def expectedTimeToComputeNIterations(n: Double): Double = {
    processingTime*n
  }

  def avgCost: Double = {
    if (costs.nonEmpty) costs.sum/costs.length.toDouble
    else 0d
  }

  def cost: Double = {
    if (costs.nonEmpty) costs.max
    else 0d
  }
}
