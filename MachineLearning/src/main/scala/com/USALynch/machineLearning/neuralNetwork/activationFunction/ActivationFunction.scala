package com.USALynch.machineLearning.neuralNetwork.activationFunction

/**
  * Reading on Activation Functions in General:
  *   https://medium.com/the-theory-of-everything/understanding-activation-functions-in-neural-networks-9491262884e0
  *   https://ml-cheatsheet.readthedocs.io/en/latest/activation_functions.html
  *
  * @author Tyler T. Lynch
  */
abstract class ActivationFunction {

  /**
    * Bounds of the function. This is needed to ensure data is normalized.
    * Along with determining activation
    *
    * If say, there is no lower bound, then use `Double.MinValue`
    * If say, there is no upper bound, then use `Double.MaxValue`
    *
    * @return (lowerBound, upperBound)
    */
  val bounds: (Double, Double) = (Double.MinValue, Double.MaxValue)

  /**
    * If you wish to specify the lower bound manually, here you go.
    * By default, it will be the value defined in bounds.
    */
  def lowerBound: Double = bounds._1

  /**
    * If you wish to specify the upper bound manually, here you go.
    * By default, it will be the value defined in bounds.
    */
  def upperBound: Double = bounds._2

  /**
    * Compute the value of x.
    * This is used in feeding values forward
    */
  def compute(x: Double): Double

  /**
    * Compute the derivative value of x.
    * This is used in back propagating values
    */
  def derivative(x: Double): Double

  /**
    * Determine which bound the value is closest to, and use that as the result.
    * @param y an input to normalize.
    */
  def round(y: Double): Double = {
    if (Math.abs(y-lowerBound) < Math.abs(y-upperBound)) lowerBound
    else upperBound
  }

  /**
    * To determine if a value is within the bounds of the function.
    * This is needed for normalizing data before it enters the Network
    */
  def withinBounds(y: Double): Boolean = {
    y >= lowerBound && y <= upperBound
  }

  /**
    * To determine if a given computed value is active or not.
    */
  def isActive(y: Double): Boolean = {
    round(y) == upperBound
  }

}
