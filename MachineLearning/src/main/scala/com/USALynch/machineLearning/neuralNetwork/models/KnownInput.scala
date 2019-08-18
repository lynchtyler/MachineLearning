package com.USALynch.machineLearning.neuralNetwork.models

import ai.x.play.json.Jsonx
import play.api.libs.json.OFormat

/**
  * A Tuple of data that the Neural Network will be able to train or validate with
  *
  * @param input a given input to be processed
  * @param result a known result for the input, or the desired outcome
  * @author Tyler T. Lynch
  */
case class KnownInput(
  input: Seq[Double],
  result: Seq[Double]
)

object KnownInput {
  implicit val format: OFormat[KnownInput] = Jsonx.formatCaseClassUseDefaults[KnownInput]
}
