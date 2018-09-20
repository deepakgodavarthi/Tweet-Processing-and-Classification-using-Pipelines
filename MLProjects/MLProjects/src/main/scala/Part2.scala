import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer}
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.{HashingTF, Tokenizer}
//import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer}
import org.apache.spark.ml.feature.StopWordsRemover
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.feature.StopWordsRemover
import org.apache.spark.mllib.clustering.LDA
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.clustering.LDA
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD

import scala.collection.mutable
object Part2 {

  def main(args: Array[String]): Unit = {
    val sc = new SparkContext(new SparkConf().setAppName("topicModel"))
    val spark = SparkSession.builder().getOrCreate()
    //val sc = new SparkContext(new SparkConf().setAppName("Hello"))
    val sqlContext = new org.apache.spark.sql.SQLContext(sc)
    import sqlContext.implicits._

    val df = spark.read.option("header", "true").option("inferSchema", "true").csv(args(0))
    df.select("airline_sentiment").distinct.count()

    val df1 = df.filter($"text".isNotNull).toDF()
    df1.select("airline_sentiment").distinct.count()

    df1.select("airline_sentiment").distinct.show()

    val df2 = df1.withColumn("airline_sentiment", when(col("airline_sentiment") === "positive", 5.0).otherwise(col("airline_sentiment"))).withColumn("airline_sentiment", when(col("airline_sentiment") === "negative", 1.0).otherwise(col("airline_sentiment"))).withColumn("airline_sentiment", when(col("airline_sentiment") === "neutral", 2.5).otherwise(col("airline_sentiment")))
    df2.select("airline_sentiment").count()

    val df3 = df2.groupBy("airline").agg(avg("airline_sentiment"))

    var output = ""

    val df4 = df3.orderBy($"avg(airline_sentiment)".desc).toDF()
    val bestAirline = df4.first.getString(0)
    val bestAirlineRating = df4.first.getDouble(1)

    output += "The best among the tweeted airlines is "+bestAirline+" with an average rating of "+bestAirlineRating+"\n"

    val df5 = df3.orderBy($"avg(airline_sentiment)").toDF()
    val worstAirline = df5.first.getString(0)
    val worstAirlineRating = df5.first.getDouble(1)

    output += "The worst among the tweeted airlines is "+worstAirline+" with an average rating of "+worstAirlineRating+"\n"

    output +="\n\n"

    output += "The topic modelling on the tweets for the best and worst airlines is as follows: \n"

    val df6 = df2.filter($"airline" === bestAirline || $"airline" === worstAirline).select("text")
    df6.count()

    val airlineTweets = sc.textFile(args(0))
    val airlineTweetsHeader = airlineTweets.first
    val airlineTweetsData = airlineTweets.filter(line => line != airlineTweetsHeader && line.split(",").length > 10)
    val airlineTweetsPairData = airlineTweetsData.map(x => ((x.split(",")) (5), (x.split(",")) (10)))

    val dataforBestWorstAirlines = airlineTweetsPairData.filter(x => (x._1 == bestAirline || x._1 == worstAirline))

    val corpus = dataforBestWorstAirlines.map(x => x._2)

    val stopWordSet = StopWordsRemover.loadDefaultStopWords("english").toSet
    val tokenized: RDD[Seq[String]] =
      corpus.map(_.toLowerCase.split("\\s")).map(_.filter(_.length > 3).filter(token => !stopWordSet.contains(token)).filter(_.forall(java.lang.Character.isLetter)))

    // Choose the vocabulary.
    //   termCounts: Sorted list of (term, termCount) pairs
    val termCounts: Array[(String, Long)] =
    tokenized.flatMap(_.map(_ -> 1L)).reduceByKey(_ + _).collect().sortBy(-_._2)
    //   vocabArray: Chosen vocab (removing common terms)
    val numStopwords = 20
    val vocabArray: Array[String] =
      termCounts.takeRight(termCounts.size - numStopwords).map(_._1)
    //   vocab: Map term -> term index
    val vocab: Map[String, Int] = vocabArray.zipWithIndex.toMap


    import scala.collection.generic._
    import scala.collection.mutable.HashMap
    val documents: RDD[(Long, Vector)] =
      tokenized.zipWithIndex.map { case (tokens, id) =>
        val counts = new mutable.HashMap[Int, Double]()
        tokens.foreach { term =>
          if (vocab.contains(term)) {
            val idx = vocab(term)
            counts(idx) = counts.getOrElse(idx, 0.0) + 1.0
          }
        }
        (id, Vectors.sparse(vocab.size, counts.toSeq))
      }

    val numTopics = 4
    val lda = new LDA().setK(numTopics).setMaxIterations(10)

    val ldaModel = lda.run(documents)

    // Print topics, showing top-weighted 10 terms for each topic.
    val topicIndices = ldaModel.describeTopics(maxTermsPerTopic = 15)

    topicIndices.foreach { case (terms, termWeights) =>
      output += "TOPIC:\n"
      terms.zip(termWeights).foreach { case (term, weight) =>
        output += {
          vocabArray(term.toInt)
        }
        output += "\t" + weight + "\n"
      }
      output += "\n\n"

    }
    sc.parallelize(List(output)).saveAsTextFile(args(1))
  }
}




