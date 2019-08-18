package com.USALynch.machineLearning.neuralNetwork.Exceptions

/**
  * The known output is of an invalid size. This prevents the network from being able to process
  * malformed data
  *
  * @author Tyler T. Lynch
  */
class InvalidOutcomeSize(given: Double, expected: Double)
  extends Exception(s"Invalid desired outcome dimensions. You input was of size ($given) and the expected size is ($expected) ")
