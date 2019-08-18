package com.USALynch.machineLearning.neuralNetwork.models

import com.USALynch.machineLearning.neuralNetwork.costFunction.CostFunction

case class Classification(
  result: Seq[Double] = Seq.empty[Double],
  activations: Seq[NetworkLayer] = Seq.empty[NetworkLayer],
  version: Option[String] = None,
  networkId: Option[String] = None,
  costFunction: Option[CostFunction] = None
)
