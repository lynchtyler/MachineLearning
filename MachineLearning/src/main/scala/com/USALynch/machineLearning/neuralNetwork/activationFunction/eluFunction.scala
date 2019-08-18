package com.USALynch.machineLearning.neuralNetwork.activationFunction

/**
  * Reading on the ELU function:
  *   https://medium.com/tinymind/a-practical-guide-to-relu-b83ca804f1f7
  *   https://ml-cheatsheet.readthedocs.io/en/latest/activation_functions.html
  *
  * @author Tyler T. Lynch
  */
case class eluFunction() extends ActivationFunction {

  override val bounds: (Double, Double) = (-1.0d, Double.MaxValue)

  /**
    * If you would like a custom E, for more precision, you are welcome to extend the LogisticFunction and override that.
    */
  lazy val E: Double = Math.E

  /**
    * Compute the value of x.
    * This is used in feeding values forward
    */
  def compute(x: Double): Double = {
    if (x < 0.0d) {
      Math.pow(E, x)-1.0d
    } else Math.max(lowerBound, x)
  }

  /**
    * Compute the derivative value of x.
    * This is used in back propagating values
    *
    * https://www.wolframalpha.com/input/?i=derive+x
    */
  def derivative(x: Double): Double = {
    if (x < 0.0d) {
      Math.pow(E, x)
    } else 1.0d
  }


  /**
    * To ensure things are working for is Active, a custom activation function
    */
  override def round(y: Double): Double = {
    if (y <= 0) lowerBound
    else upperBound
  }

}
