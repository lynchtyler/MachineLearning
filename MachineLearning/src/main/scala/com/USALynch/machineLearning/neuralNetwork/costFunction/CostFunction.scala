package com.USALynch.machineLearning.neuralNetwork.costFunction


/**
  * Reading on Error/Cost Functions in General:
  *   https://stats.stackexchange.com/questions/154879/a-list-of-cost-functions-used-in-neural-networks-alongside-applications
  *   https://hlab.stanford.edu/brian/error_sum_of_squares.html
  *   https://www.cs.cmu.edu/~atalwalk/teaching/winter17/cs260/lectures/lec08.pdf
  *
  * A video on costs:
  *   https://youtu.be/IHZwWFHWa-w?t=555
  *
  * @author Tyler T. Lynch
  */
abstract case class CostFunction(
  results: Seq[Double] = Seq.empty,
  desiredOutcome: Seq[Double] = Seq.empty
) {

  def averageCost: Double = {
    if (results.nonEmpty) results.sum/results.length else 0d
  }

  def costs: Seq[Double]

  def derivatives: Seq[Double]

  def cost(result: Double, desiredOutcome: Double): Double

  def derivative(result: Double, desiredOutcome: Double): Double

}

object CostFunction {

  def avgGradient(costs: Seq[CostFunction]): Seq[Double] = {
    val allCosts = costs.map(_.costs).flatMap(_.zipWithIndex).groupBy(_._2)
    allCosts.map{ case (_, groupedCosts) =>
      val sum = groupedCosts.map(_._1).sum
      if (groupedCosts.nonEmpty) sum/groupedCosts.length.toDouble else 0d
    }.toSeq
  }
}