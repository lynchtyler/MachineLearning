import sbt._
import play.sbt.PlayImport.guice


object Dependencies {


  val breezeVersion = "0.13.2"
  val breeze = "org.scalanlp" %% "breeze" % breezeVersion
  val breezeNatives = "org.scalanlp" %% "breeze-natives" % breezeVersion
  val breezeViz = "org.scalanlp" %% "breeze-viz" % breezeVersion

  val breezeDependencies: Seq[ModuleID] = Seq(
    breeze, breezeNatives, breezeViz
  )


  val netlibVersion = "1.1"
  val netlibCoreVersion = "1.1.2"
  val netlibCore = "com.github.fommil.netlib" % "core" % netlibCoreVersion
  val allNetLib = "com.github.fommil.netlib" % "all" % netlibCoreVersion
  val netlib: Option[ModuleID] = System.getProperty("os.name").toLowerCase match {
    case mac if mac.contains("mac")  => Some("com.github.fommil.netlib" % "netlib-native_system-osx-x86_64" % netlibVersion)
    case linux if linux.contains("linux") => Some("com.github.fommil.netlib" % "netlib-native_system-linux-x86_64" % netlibVersion)
    case win if win.contains("win") => {
      // Windows isn't quite supported for breeze.
      None
    }
    case osName => throw new RuntimeException(s"Unknown operating system $osName")
  }
  // https://mxnet.incubator.apache.org/
  val mxMax = "org.apache.mxnet" % "mxnet-full_2.11-linux-x86_64-gpu" % "1.4.1"

  // https://github.com/eaplatanios/tensorflow_scala
  val tensorFlowScala = "org.platanios" %% "tensorflow-api" % "0.4.1"

  val deepLearning = "com.thoughtworks.deeplearning" %% "deeplearning" % "2.2.0-M1"
  val computeVersion = "0.4.2"
  val computeTensors = "com.thoughtworks.compute" %% "tensors" % computeVersion
  val computeOpenCl = "com.thoughtworks.compute" %% "opencl" % computeVersion
  val computeMemory = "com.thoughtworks.compute" %% "memory" % computeVersion
  val computeScala = "com.thoughtworks.compute" %% "compute-scala" % "0.1.1"

   // https://github.com/ThoughtWorksInc/Compute.scala
  val computeCpu = "com.thoughtworks.compute" %% "cpu" % computeVersion
  val computeGpu = "com.thoughtworks.compute" %% "gpu" % computeVersion
  val lwjglVersion = "3.2.1"
  val lwjgl = ("org.lwjgl" % "lwjgl" % lwjglVersion)
  .jar().classifier {
    import scala.util.Properties._
    if (isMac) {
      "natives-macos"
    } else if (isLinux) {
      "natives-linux"
    } else if (isWin) {
      "natives-windows"
    } else {
      throw new MessageOnlyException(s"lwjgl does not support $osName")
    }
  }
  // LWJGL OpenCL library
  val lwjglOpenClBindings = "org.lwjgl" % "lwjgl-opencl" % lwjglVersion
  val lwjglOpenCl = "org.lwjgl.osgi" % "org.lwjgl.opencl" % lwjglVersion

  val nd4jVersion = "0.9.1"
  val nd4j = "org.nd4j" % "nd4j-api" % nd4jVersion
  val nd4jNative = "org.nd4j" % "nd4j-native" % nd4jVersion

  val linearAlgebraDependencies: Seq[ModuleID] = Seq(
    netlibCore, allNetLib, tensorFlowScala
  ) ++ netlib.toSeq

  val computeDependencies: Seq[ModuleID] = Seq(
    // computeTensors, computeMemory, computeScala,

    // computeOpenCl,

    computeCpu,
    computeGpu,
    lwjgl, lwjglOpenCl, lwjglOpenClBindings,

    deepLearning, nd4j, nd4jNative
  ) ++ linearAlgebraDependencies


  val scala_2_11_LinearAlgebraDependencies = Seq(
    mxMax
  )

  val akkaVersion = "2.5.21"
  val akkaActor = "com.typesafe.akka" %% "akka-actor" % akkaVersion
  val akkaStream = "com.typesafe.akka" %% "akka-stream" % akkaVersion
  val akkaCluster = "com.typesafe.akka" %% "akka-cluster" % akkaVersion
  val akkaHttp = "com.typesafe.akka" %% "akka-http" % "10.1.7"
  val reactiveKafkaVersion = "1.0"
  val akkaStreamKafka = "com.typesafe.akka" %% "akka-stream-kafka" % reactiveKafkaVersion


  val akkaDependencies = Seq(
    akkaActor, akkaStream, akkaHttp, akkaCluster, akkaStreamKafka
  )



  val playJson = "com.typesafe.play" %% "play-json" % "2.7.4"


  val mathCommons = "org.apache.commons" % "commons-math3" % "3.6.1"


  val machineLearningScalaVersion = "2.12.0"
  val machineLearningDependencies: Seq[ModuleID] = Seq(
    "org.scalatestplus.play" %% "scalatestplus-play" % "3.1.2" % Test,
    "joda-time" % "joda-time" % "2.10",
    "ai.x" %% "play-json-extensions" % "0.20.0",
    // https://mvnrepository.com/artifact/org.mongodb.scala/mongo-scala-driver
    // For the ObjectId stuffs
    "org.mongodb.scala" %% "mongo-scala-driver" % "2.6.0",
    playJson, mathCommons
  ) ++ computeDependencies ++ breezeDependencies ++ akkaDependencies



}
