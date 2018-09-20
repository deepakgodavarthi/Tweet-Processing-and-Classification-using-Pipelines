import org.apache.spark.sql.SparkSession
val spark = SparkSession.builder().getOrCreate()



val df = spark.read.option("header","true").option("inferSchema","true").csv("Tweets.csv")
df.columns
df.select("airline_sentiment").show()
