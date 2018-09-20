import org.apache.spark.mllib.fpm.FPGrowth
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.functions._
object Part3 {
  def main(args: Array[String]): Unit = {
    val sc = new SparkContext(new SparkConf().setAppName("Part3"))
    val spark = SparkSession.builder().getOrCreate()
    val data = sc.textFile(args(0))
    val header = data.first
    val data1 = data.filter(line => line != header)
    val transactions = data1.map(x => ((x.split(",")) (0), (x.split(",")) (1)))

    transactions.take(10)
    val tlist = transactions.groupByKey.mapValues(_.toArray)
    val tmpLst = tlist.map(x => x._2)
    val fpg = new FPGrowth().setMinSupport(0.001).setNumPartitions(6)
    val model = fpg.run(tmpLst)
    var output = ""
    output+="Top 10 frequent items: \n"
    model.freqItemsets.sortBy(-_.freq).take(10).foreach { itemset =>
      output+=itemset.items.mkString("[", ",", "]")
      output += ", " + itemset.freq+"\n"
    }
    output+="\n\n"
    output+="Association Rules: \n"
    val minConfidence = 0.5
    model.generateAssociationRules(minConfidence).sortBy(-_.confidence).take(10).foreach { rule =>
      output += rule.antecedent.mkString("[", ",", "]") + " => "
      output += rule.consequent.mkString("[", ",", "]") + ", " + rule.confidence+"\n"
    }
    sc.parallelize(List(output)).saveAsTextFile(args(1))
  }
}

