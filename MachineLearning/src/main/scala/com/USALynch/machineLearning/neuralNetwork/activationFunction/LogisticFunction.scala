package com.USALynch.machineLearning.neuralNetwork.activationFunction

/**
  * Reading on the Logistic/Sigmoid function:
  *   http://mathworld.wolfram.com/SigmoidFunction.html
  *   https://en.wikipedia.org/wiki/Logistic_function
  *   https://en.wikipedia.org/wiki/Sigmoid_function
  *
  *
  * As the growthRate approaches infinity, the faster the value will increase in value as x increases
  *   or conversely, the faster the value will decrease in value as x decreases
  * As the growthRate approaches 0, the slower the value will increase in value as x increases
  *   or conversely, the slower the value will decrease in value as x decreases
  *
  * @param growthRate the growth Rate will modify the rate at which the value increases/decrease
  * @param midPointX the x value of the midpoint. If you would like to offset that some bias, you are welcome to.
  *
  * @author Tyler T. Lynch
  */
case class LogisticFunction(
  growthRate: Double = 1.0d,
  midPointX: Double = 0.0d
) extends ActivationFunction {

  override val bounds: (Double, Double) = (0d, 1d)

  /**
    * If you would like a custom E, for more precision, you are welcome to extend the LogisticFunction and override that.
    */
  lazy val E: Double = Math.E

  /**
    * Compute the value of x.
    * This is used in feeding values forward
    */
  def compute(x: Double): Double = {
    val l = upperBound
    val expo = -(growthRate * (x - midPointX))
    val denominator = 1 + Math.pow(E, expo)
    l/denominator
  }

  /**
    * Compute the derivative value of x.
    * This is used in back propagating values
    *
    * Some good documentation on this:
    *   https://www.anotsorandomwalk.com/first-derivative-of-the-sigmoid-function/
    *   https://math.stackexchange.com/questions/78575/derivative-of-sigmoid-function-sigma-x-frac11e-x
    *   Step 4: https://www.anotsorandomwalk.com/backpropagation-example-with-numbers-step-by-step/
    */
  def derivative(x: Double): Double = {
    x*(1.0d-x)
  }



}
