package com.USALynch.machineLearning.neuralNetwork.activationFunction

/**
  * Reading on the ReLU function:
  *   https://medium.com/tinymind/a-practical-guide-to-relu-b83ca804f1f7
  *   https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
  *   https://ml-cheatsheet.readthedocs.io/en/latest/activation_functions.html
  *
  * @author Tyler T. Lynch
  */
case class ReluFunction(override val upperBound: Double = 10.0d) extends ActivationFunction {

  override val bounds: (Double, Double) = (0d, this.upperBound)

  /**
    * Compute the value of x.
    * This is used in feeding values forward
    */
  def compute(x: Double): Double = {
    Math.max(lowerBound, x)
  }

  /**
    * Compute the derivative value of x.
    * This is used in back propagating values
    *
    * https://www.wolframalpha.com/input/?i=derive+x
    */
  def derivative(x: Double): Double = {
    1.0d
  }


  /**
    * To ensure things are working for is Active, a custom activation function
    */
  override def round(y: Double): Double = {
    if (y <= 0) lowerBound
    else upperBound
  }

}
