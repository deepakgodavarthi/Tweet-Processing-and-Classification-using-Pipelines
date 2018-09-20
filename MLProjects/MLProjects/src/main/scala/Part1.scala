import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer}
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.{HashingTF, Tokenizer}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.Row
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
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.IntegerType
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
object Part1 {
  def main(args: Array[String]): Unit ={
    if (args.length == 0) {
      println("I need two parameters ")
    }


    val sc = new SparkContext(new SparkConf().setAppName("Part1"))
    val spark = SparkSession.builder().getOrCreate()
    val sqlContext = new org.apache.spark.sql.SQLContext(sc)
    import sqlContext.implicits._
    var output = ""
    val df = spark.read.option("header","true").option("inferSchema","true").csv(args(0))
    val df1 = df.drop("tweet_id").drop("negativereason").drop("airline_sentiment_gold").drop("name").drop("negativereason_gold").drop("tweet_coord").drop("tweet_created").drop("tweet_location").drop("user_timezone")
    val indexer = new StringIndexer().setInputCol("airline_sentiment").setOutputCol("airline_sentiment_index")
    val df2 = indexer.fit(df1).transform(df1)
    val df3 = df2.na.fill(0, Seq("negativereason_confidence"))
    val indexer2 = new StringIndexer().setInputCol("airline").setOutputCol("airline_index")
    val df4 = indexer2.fit(df3).transform(df3)
    val df5 = df4.drop("airline_sentiment").drop("airline")
    val df6 = df5.filter($"text".isNotNull).toDF()
    val tokenizer = new Tokenizer().setInputCol("text").setOutputCol("words")
    val df7 = tokenizer.transform(df6)
    val remover = new StopWordsRemover().setInputCol("words").setOutputCol("words1")
    val df8 = remover.transform(df7)
    val hashingTF = new org.apache.spark.ml.feature.HashingTF().setInputCol("words1").setOutputCol("rawFeatures").setNumFeatures(2000)
    val df9 = hashingTF.transform(df8)
    val idf = new IDF().setInputCol("rawFeatures").setOutputCol("raw2")
    val idfModel = idf.fit(df9)
    val df10 = idfModel.transform(df9)
    val df11 = df10.drop("text").drop("words").drop("words1").drop("rawFeatures")
    val df13 = df11.withColumnRenamed( "airline_sentiment_index", "label")
    val df12 = df13.na.fill("0")
    val df14 = df12.select( df12("airline_sentiment_confidence").cast(IntegerType).as("airline_sentiment_confidence"), df12("negativereason_confidence").cast(IntegerType).as("negativereason_confidence"), df12("retweet_count"), df12("label").cast(IntegerType).as("label"), df12("airline_index").cast(IntegerType).as("airline_index"), df12("raw2"))
    val Array(train, test) =df14.randomSplit(Array(0.9,0.1))
    val assembler = new VectorAssembler()
    assembler.setInputCols(Array("airline_sentiment_confidence", "negativereason_confidence", "retweet_count", "airline_index", "raw2"))
    assembler.setOutputCol("features")
    val df15 = assembler.transform(df14)
    val lr = new LogisticRegression().setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8).setLabelCol("label").setFamily("multinomial")
    val pipeline = new Pipeline().setStages(Array(assembler,lr))
    val paramGrid = new ParamGridBuilder().addGrid(lr.threshold, Array(0.1,0.5,0.6)).addGrid(lr.regParam, Array(0.1,0.001, 0.001)).addGrid(lr.maxIter, Array(10,20,30)).build()
    //val evaluator2 = new MulticlassClassificationEvaluator().setLabelCol("label")
    //Build cross validator with 5 folds
    output+="Running Logistic Regression Classifier:\n"
    val evaluator2 = new MulticlassClassificationEvaluator()
    evaluator2.setLabelCol("label")
    val cv = new CrossValidator().setEstimator(pipeline).setEstimatorParamMaps(paramGrid).setEvaluator(evaluator2).setNumFolds(5)
    val model = cv.fit(train)
    val result = model.transform(test)
    evaluator2.setMetricName("accuracy")
    val accuracy = evaluator2.evaluate(result)
    output+="Accuracy of the model is :"+accuracy+"\n"
    evaluator2.setMetricName("weightedPrecision")
    val weightedPrecision = evaluator2.evaluate(result)
    output+="Weighted Precision of the model is :"+weightedPrecision+"\n"
    evaluator2.setMetricName("weightedRecall")
    val weightedRecall = evaluator2.evaluate(result)
    output+="Weighted Recall of the model is :"+weightedRecall+"\n"
    output+="\n\n"
    output+="Running random forest classifier:\n"
    val rf = new RandomForestClassifier().setLabelCol("label").setFeaturesCol("features").setNumTrees(50).setMaxDepth(20)
    val pipeline_rf = new Pipeline().setStages(Array(assembler,rf))
    //Build Paramgrid Builder for Random Forest Classifier
    val paramGrid_rf = new ParamGridBuilder().addGrid(rf.numTrees, Array(20,35,50)).addGrid(rf.maxDepth, Array(10,15,20)).build()
    val evaluator1 = new MulticlassClassificationEvaluator()
    evaluator1.setLabelCol("label")
    val cv_rf = new CrossValidator().setEstimator(pipeline_rf).setEstimatorParamMaps(paramGrid_rf).setEvaluator(evaluator1).setNumFolds(5)
    val model_rf = cv_rf.fit(train)
    val result_rf = model_rf.transform(test)
    evaluator1.setMetricName("accuracy")
    val accuracy1 = evaluator1.evaluate(result_rf)
    output+="Accuracy of the model is :"+accuracy1+"\n"
    evaluator1.setMetricName("weightedPrecision")
    val weightedPrecision1 = evaluator1.evaluate(result_rf)
    output+="Weighted Precision of the model is :"+weightedPrecision1+"\n"
    evaluator1.setMetricName("weightedRecall")
    val weightedRecall1 = evaluator1.evaluate(result_rf)
    output+="Weighted Recall of the model is :"+weightedRecall1+"\n"
    sc.parallelize(List(output)).saveAsTextFile(args(1))
  }
}




