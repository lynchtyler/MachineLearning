package com.USALynch.machineLearning.neuralNetwork.models

import ai.x.play.json.Jsonx
import play.api.libs.json.OFormat

import scala.util.Random

/**
  * To keep training data and validation data separate
  *
  * @param trainingData data that will be used for training on
  * @param validationData data that will be used to validate the training with
  *
  * @author Tyler T. Lynch
  */
case class ValidationData(
  trainingData: Seq[KnownInput] = Seq.empty[KnownInput],
  validationData: Seq[KnownInput] = Seq.empty[KnownInput]
)

object ValidationData {

  implicit val format: OFormat[ValidationData] = Jsonx.formatCaseClassUseDefaults[ValidationData]

  def createValidationData(input: Seq[KnownInput], percentageToValidate: Double): ValidationData = {
    val r = Random
    val shuffled = r.shuffle(input)
    val len = shuffled.length
    val numberOfItemsToTrain = (len * (1.0 - percentageToValidate)).toInt
    if (input.nonEmpty) ValidationData(shuffled.take(numberOfItemsToTrain), shuffled.drop(numberOfItemsToTrain))
    else ValidationData()
  }
}
