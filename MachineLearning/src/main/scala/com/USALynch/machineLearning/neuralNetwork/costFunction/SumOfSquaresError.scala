package com.USALynch.machineLearning.neuralNetwork.costFunction

/**
  * Reading on SSE
  *   https://hlab.stanford.edu/brian/error_sum_of_squares.html
  *
  * A video on costs:
  *   https://youtu.be/IHZwWFHWa-w?t=555
  *
  * @author Tyler T. Lynch
  */
class SumOfSquaresError(
  override val results: Seq[Double] = Seq.empty,
  override val desiredOutcome: Seq[Double] = Seq.empty) extends CostFunction() {

  def costs: Seq[Double] = {
    (results zip desiredOutcome).map{ case (r, desired) =>
      cost(r, desired)
    }
  }

  def derivatives: Seq[Double] = {
    (results zip desiredOutcome).map{ case (r, desired) =>
      derivative(r, desired)
    }
  }

  def cost(result: Double, desiredOutcome: Double): Double = {
    // * 0.5 to make the derivative calculation clean
    Math.pow(result - desiredOutcome, 2.0d) * 0.5
  }

  def derivative(result: Double, desiredOutcome: Double): Double = {
    result - desiredOutcome
  }

}
