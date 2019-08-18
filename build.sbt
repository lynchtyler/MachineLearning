import sbt.Keys.{javaOptions, libraryDependencies, _}


lazy val commonSettings = Seq(
  organization := "com.USALynch",
  scalaVersion := "2.12.6", // Cannot update this value if using reactive mongo
  crossScalaVersions := Seq("2.11.12", "2.12.7"),
  version := "1.0-SNAPSHOT",
  javaOptions += "-Xmx8gb", // 8gb should hopefully be enough....
  resolvers ++= Seq(
    Resolver.sonatypeRepo("snapshots"),
    "Sonatype Snapshots" at "https://oss.sonatype.org/content/repositories/snapshots/",
    "Typesafe Releases" at "https://repo.typesafe.com/typesafe/maven-releases/",
    "Maven central" at "http://repo1.maven.org/maven2/",
    "Sonatype Releases" at "https://oss.sonatype.org/content/repositories/releases/",
    "scalac repo" at "https://raw.githubusercontent.com/ScalaConsultants/mvn-repo/master/"
  )
)


lazy val machineLearning = (project in file("./MachineLearning"))
  .settings(commonSettings: _*)
  .settings(
    name := s"""MachineLearning""",
    libraryDependencies ++= Dependencies.machineLearningDependencies
  )

