package final2.final2

import org.apache.spark.SparkContext._
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import edu.stanford.nlp.pipeline._
import edu.stanford.nlp.util._
import edu.stanford.nlp.ling.CoreAnnotations._
import scala.collection.JavaConversions._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.SQLContext
import org.apache.hadoop.conf.Configuration
import org.apache.spark.ml.feature.CountVectorizer
import org.apache.spark.ml.feature.IDF
import org.apache.spark.mllib.linalg.{Vectors, Vector => MLLibVector}
import org.apache.spark.ml.linalg.{Vector => MLVector}
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.linalg.{Matrix, SingularValueDecomposition}
import scala.collection.mutable.ArrayBuffer
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.{Word2Vec, Word2VecBase, Word2VecModel, VectorAssembler, OneHotEncoder}
import org.apache.spark.ml.classification.BinaryLogisticRegressionSummary
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import com.databricks.spark.csv._
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.Row


object fin {
  
  val sc = new SparkContext(new SparkConf().setAppName("Cupid Analytics").setMaster("local[*]").set("spark.driver.memory","2g"))
  val sqlContext = new SQLContext(sc)
  import sqlContext.implicits._
  def main(args: Array[String]) = {
    val data : Dataset[LemmaTuple]= sqlContext.read.textFile("hdfs://quickstart.cloudera:8020/user/cloudera/final/gtd_f/part-m-00000").map({
        text => {
          val tuple = mkAttackTuple(text)
          val pipe = startPipe()
          
          new LemmaTuple(tuple.iyear, tuple.imonth, tuple.country, tuple.success, tuple.attackType, tuple.nkill, getLemmas(tuple.source1,pipe), getLemmas(tuple.source2,pipe))
        }
      })
      val terrorDF = data.toDF("iyear","imonth","country","success","attackType","nkill","source1","source2")
      terrorDF.cache()
      val filteredDF = terrorDF.filter(not($"iyear" === "iyear"))
        .filter(not($"iyear" === "BR"))
        .select("iyear","imonth","country","success","attackType","nkill","source1","source2")
      filteredDF.show()
      filteredDF.cache()
      /*
      * +-----+------+-------------+-------+--------------------+-----+
        |iyear|imonth|      country|success|          attackType|nkill|
        +-----+------+-------------+-------+--------------------+-----+
        | 2003|     1|       Russia|      1|       Armed Assault|    1|
        | 2003|     1|     Colombia|      1|   Bombing/Explosion|    0|
        | 2003|     1|     Colombia|      1|   Bombing/Explosion|    0|
        | 2003|     1|     Colombia|      1|   Bombing/Explosion|    0|
        | 2003|     1|     Colombia|      1|   Bombing/Explosion|    0|
        | 2003|     1|     Colombia|      1|   Bombing/Explosion|    0|
        | 2003|     1|     Colombia|      1|   Bombing/Explosion|    0|
        | 2003|     1|     Colombia|      1|   Bombing/Explosion|    0|
        | 2003|     1|     Colombia|      1|   Bombing/Explosion|    0|
        | 2003|     1|     Colombia|      1|   Bombing/Explosion|    0|
        | 2003|     1|     Colombia|      1|   Bombing/Explosion|    0|
        | 2003|     1|     Colombia|      1|   Bombing/Explosion|    0|
        | 2003|     1|        Nepal|      1|       Armed Assault|    0|
        | 2003|     1|  Philippines|      1|   Bombing/Explosion|    1|
        | 2003|     1|    Indonesia|      0|       Armed Assault|    0|
        | 2003|     1|United States|      1|Facility/Infrastr...|    0|
        | 2003|     1|       Uganda|      1|Hostage Taking (K...|    0|
        | 2003|     1|       Uganda|      1|       Armed Assault|    0|
        | 2003|     1|        India|      1|       Assassination|    1|
        | 2003|     1|        India|      1|       Assassination|    1|
        +-----+------+-------------+-------+--------------------+-----+
       */
      val siYear = new StringIndexer().setInputCol("iyear").setOutputCol("lblYear")
      val siMonth = new StringIndexer().setInputCol("imonth").setOutputCol("lblMonth")
      val siCountry = new StringIndexer().setInputCol("country").setOutputCol("lblCountry")
      val siSuccess = new StringIndexer().setInputCol("success").setOutputCol("lblSuccess")
      val siAT = new StringIndexer().setInputCol("attackType").setOutputCol("lblAT")
      val siNK = new StringIndexer().setInputCol("nkill").setOutputCol("label")
      val ixYear = new OneHotEncoder()
        .setInputCol("lblYear")
        .setOutputCol("ixYear")
      val ixMonth = new OneHotEncoder()
        .setInputCol("lblMonth")
        .setOutputCol("ixMonth")
      val ixCountry = new OneHotEncoder()
        .setInputCol("lblCountry")
        .setOutputCol("ixCountry")
      val ixSuccess = new OneHotEncoder()
        .setInputCol("lblSuccess")
        .setOutputCol("ixSuccess")
      val ixAT = new OneHotEncoder()
        .setInputCol("lblAT")
        .setOutputCol("ixAT")
        
      val labeledFrame = siNK.fit(terrorDF).transform(terrorDF)
      
      val termLimit = 10000
      val countVectorizer = new CountVectorizer().setInputCol("source1").setOutputCol("source1Freqs").setVocabSize(termLimit)
      val countVectorizer1 = new CountVectorizer().setInputCol("source2").setOutputCol("source2Freqs").setVocabSize(termLimit)
      val idf = new IDF().setInputCol(countVectorizer.getOutputCol).setOutputCol("idf")
      val idf1 = new IDF().setInputCol(countVectorizer1.getOutputCol).setOutputCol("idf1")
      val w2v = new Word2Vec().setInputCol("source1").setOutputCol("w2v")
      val w2v1 = new Word2Vec().setInputCol("source2").setOutputCol("w2v1")
        
      val va = new VectorAssembler().setInputCols(Array("ixYear","ixMonth","ixCountry","ixSuccess","ixAT","idf","idf1","w2v","w2v1")).setOutputCol("features")
      
      
      val pipeline = new Pipeline().setStages(Array(siYear,siMonth, siCountry, siSuccess, siAT, siNK, ixYear, ixMonth, ixCountry, ixSuccess, ixAT, countVectorizer,countVectorizer1,idf,idf1, w2v, w2v1, va))
      val fmtDF = pipeline.fit(filteredDF).transform(filteredDF)
      val seed = 20564
      val Array(training, test) = fmtDF.randomSplit(Array(0.7, 0.3), seed)
      val lr = new LogisticRegression().setFeaturesCol("features").setLabelCol("label")
      val lrDF = lr.fit(training).transform(test)
      lrDF.select("iyear","imonth", "probability", "label", "prediction", "nkill")
      .collect()//converts the dataframe into an array of Row objects
      .foreach{
      case Row(iyear: String, imonth: String, probability: MLVector, label: Double, prediction: Double, nkill: String) => 
        println(s"($iyear,$imonth, $nkill, $label) --> prob=$probability, pred=$prediction")
          }
      
    val lp = lrDF.select( "label", "prediction")
      val counttotal = lrDF.count()
      val correct = lp.filter($"label" === $"prediction").count()
      val wrong = lp.filter(not($"label" === $"prediction")).count()
      val truep = lp.filter($"prediction" === 1.0).filter($"label" === $"prediction").count() 
      val truen = lp.filter($"prediction" === 0.0).filter($"label" === $"prediction").count()
      val falseN = lp.filter($"prediction" === 1.0).filter(not($"label" === $"prediction")).count() 
      val falseP = lp.filter($"prediction" === 0.0).filter(not($"label" === $"prediction")).count() 
      val ratioWrong=wrong.toDouble/counttotal.toDouble
      val ratioCorrect=correct.toDouble/counttotal.toDouble
      val precision = truep.toDouble/counttotal.toDouble
      val recall = truep.toDouble/(truep.toDouble+falseN.toDouble)
      println(s"accuracy: ${ratioCorrect}")
      println(s"correct: ${correct}")
      println(s"wrong: ${wrong}")
      println(s"Total Count: ${counttotal}")
      println(s"true positive count: ${truep}")
      println(s"false negative count: ${falseN}")
      println(s"false positive count: ${falseP}")
      println(s"precision ratio: ${precision}")
      println(s"recall ratio: ${recall}")
  }
  
  
  def startPipe(): StanfordCoreNLP = {
    val pipe = new StanfordCoreNLP(PropertiesUtils.asProperties(
        "annotators", "tokenize,ssplit,pos,lemma"))
    pipe
  }
  
  def isLetters (str: String): Boolean = {
    str.forall(x => Character.isLetter(x))
  }
  
  def getLemmas(text: String, cNLP: StanfordCoreNLP): Seq[String] = {
    
    val document = new Annotation(text)
    cNLP.annotate(document)
    val lemmas = new scala.collection.mutable.ArrayBuffer[String]()
    
    val sentences = document.get(classOf[SentencesAnnotation])
    for (sentence <- sentences; 
    token <- sentence.get(classOf[TokensAnnotation])) {
      val lemma = token.get(classOf[LemmaAnnotation])
      if (lemma.length > 2 &&  isLetters(lemma)) {
        lemmas += lemma.toLowerCase 
      }
    }
    lemmas
  }
  //initial preprocessor to extract fields from the comma separated data
  def mkAttackTuple(line: String) : TerrorCase = {
    val terrorFields = line.split(",(?=([^\"]*\"[^\"]*\")*[^\"]*$)")
    if (terrorFields.size == 8 )
    {
      val iyear = terrorFields.apply(0)
      val imonth = terrorFields.apply(1)
      val country = terrorFields.apply(2)
      val success = terrorFields.apply(3)
      val attackType = terrorFields.apply(4)
      val nkill = terrorFields.apply(5)
      val source1 = terrorFields.apply(6)
      val source2 = terrorFields.apply(7)
      TerrorCase(iyear, imonth, country, success, attackType, nkill,source1,source2)
    }
    else
    {
      val iyear = "BR"
      val imonth = "BR"
      val country = "BR"
      val success = "BR"
      val attackType = "BR"
      val nkill = "BR"
      val source1 = "BR"
      val source2 = "BR"
      TerrorCase(iyear, imonth, country, success, attackType, nkill, source1, source2)
    }
  }

}