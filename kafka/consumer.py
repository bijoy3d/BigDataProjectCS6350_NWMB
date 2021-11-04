import json 
from kafka import KafkaConsumer
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode, from_json
from pyspark.sql.functions import split
from ast import literal_eval

from pyspark.sql.types import FloatType, IntegerType, StringType, StructType, ArrayType


#def consumer():
if __name__ == '__main__':
    spark = SparkSession.builder.appName("StructuredKafkaWordCount").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    spark.sql("set spark.sql.caseSensitive=true")
    schema = StructType().add("T", StringType()).add("S", StringType()).add("i", IntegerType()).add("x", StringType()).add("p", FloatType()).add("s", IntegerType()).add("c", StringType()).add("z", StringType()).add("t", StringType())
    # Create DataSet representing the stream of input lines from kafka
    lines = spark.readStream.format("kafka").option("kafka.bootstrap.servers", 'localhost:9092').option('subscribe', 'messages').load().select(from_json(col("value").cast("string"), schema).alias("pdata"))#.selectExpr("CAST(value AS STRING)")
    #.selectExpr("CAST(value AS STRING)")
    #.select(from_json(col("value").cast("string"), schema).alias("pdata"))#.selectExpr("CAST(value AS STRING)")
    print("----------------------------")
    #print(lines.show())
    print(lines)
    print("----------------------------")
    #words=lines
    words = lines.select(col("pdata.S").alias('Stock Symbol'), col("pdata.p").alias("Price"))


    query = words.writeStream.format("console").option("truncate",False).start()

    query.awaitTermination()