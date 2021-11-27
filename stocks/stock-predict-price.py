import argparse
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json
from pyspark import SparkContext
from threading import Thread
from pyspark.streaming import StreamingContext
from pyspark.sql.types import StringType, StructType
import sys
import time 
import shutil
import pandas as pd
sys.path.append("..")
from lstm import LSTM 
from pyspark.sql import SQLContext
from sklearn.preprocessing import MinMaxScaler
from pyspark.sql import Row

mainDF = pd.read_csv('/home/nithyashanmugam/TradeApp/apple_5min_data_1week.csv')

class StockPrediction():
    def __init__(self):      
        dir_path = '/tmp/checkpoint'
        try:
            shutil.rmtree(dir_path)
        except OSError as e:
            print("Error: %s : %s" % (dir_path, e.strerror))
         
        self.spark = SparkSession.builder.appName("StockPrediction").config("spark.executor.memory", "70g").config("spark.driver.memory", "50g").config("spark.memory.offHeap.enabled",True).config("spark.memory.offHeap.size","16g").config("es.index.auto.create", "true").getOrCreate()
        self.spark.sparkContext.setLogLevel("ERROR")
        self.spark.conf.set("spark.sql.streaming.checkpointLocation", "/tmp/checkpoint")
        self.spark.sql("set spark.sql.caseSensitive=true")
        self.sc = self.spark.sparkContext;
        self.sqlContext = SQLContext(self.sc)	
        self.df = self.spark.read.option("inferSchema", "true").option("header", "true").csv("/home/nithyashanmugam/TradeApp/apple_5min_data_1week.csv")
        self.pdf = self.df.toPandas()        
        self.opscaler = MinMaxScaler()
        self.ipscaler = MinMaxScaler()
        print(self.df.show(10))
        self.cleanDataset()
        print(self.df.show(10))   
       
        
    def cleanDataset(self) :        
        self.df = self.df.drop("Date")         
        global mainDF        
        mainDF = mainDF.drop(["Date"], axis=1)
       

    # This topic would be parsed by logstash and Kibana to show the analysis.
    def kafka_setup(self,kafkaListenServer, listenTopic):
    
        def process_row(row):   
           mainDF = pd.read_csv('/home/nithyashanmugam/TradeApp/stream.csv')      
           print(mainDF.tail(5))           
           print("inside process row ")
           column_names = ["Open", "High", "Low","Close","Volume","Trade_count","vwap"]
           newDf = pd.DataFrame(row)   
           newDf = newDf.T
           newDf.columns = column_names           
           newDf.reset_index(drop=True, inplace=True) 
           mainNewDF = mainDF.append(newDf, ignore_index=False) 
           mainNewDF.reset_index(drop=True, inplace=True) 
           mainNewDF.to_csv('/home/nithyashanmugam/TradeApp/stream.csv', index=False)
           print(mainNewDF.tail(4))
           

        #datafrane.write to local csv file and after line 68 read the csv append to the main dataframe and predict
        
        # This function would send the Close Price to the trained model to find the predicted price.
        def prdeictStockPrice(lines):        
            selected = lines.select("Close", "PredictedPrice").withColumnRenamed("PredictedPrice","value")
            selected.writeStream.format("kafka").option("kafka.bootstrap.servers", kafkaWriteServer).option('topic', writeTopic).start()
            self.spark.streams.awaitAnyTermination()

        
        
        # The Close Price is read here from the Kafka topic to which the alpaca json was sent by the alpaca API
        #dataframe -> csv 
        schema = StructType().add("Open", StringType()).add("Low", StringType()).add("High", StringType()).add("Close", StringType()).add("Volume", StringType()).add("vwap", StringType()).add("Trade_count", StringType())
        # Create DataSet representing the stream of input lines from kafka
       
        lines = self.spark.readStream.format("kafka").option("kafka.bootstrap.servers", kafkaListenServer).option('subscribe', listenTopic).load().select(from_json(col("value").cast("string"), schema).alias("pdata"))      
        lines = lines.select(col("pdata.Open").alias('Open'),col("pdata.High").alias('High'),col("pdata.Low").alias('Low'),col("pdata.Close").alias('Close'),col("pdata.Volume").alias('Volume'),col("pdata.Trade_count").alias('Trade_count'),col("pdata.vwap").alias('vwap'))    
            
        #lines.writeStream.format("kafka").option("kafka.bootstrap.servers", kafkaWriteServer).option('topic', 'apache').start()
        #spark.streams.awaitAnyTermination()
        lines = lines.writeStream.format("memory").foreach(process_row).option("checkpointLocation", "/tmp/checkpoint/").outputMode("append").start()          
        lines.awaitTermination()
        
def main():     
    parser = argparse.ArgumentParser(description="Run Application to see Predicted Price for the stocks")
    parser.add_argument('-m','--modelFile', help='Path on local drive where to the save trained Model if -b is enabled or just load the trained model from', default='/home/nithyashanmugam/TradeApp/nnmodel')
    parser.add_argument('-b','--buildModel', help='Option to specify to build the model and path on local drive to CSV File. If not enabled then use the trained model stored in -m', action='store_true')
    parser.add_argument('-c','--csvFile', help='Path to CSV File where training data is saved. This would be used if -b is enabled', default="/home/nithyashanmugam/TradeApp/apple_5Min_data.csv")
    parser.add_argument('-kl','--kafkaListenServer', help='Kafka Server:PORT from where we are listening for data', default="localhost:9092")
    parser.add_argument('-kw','--kafkaWriteServer', help='Kafka Server:PORT to where we are writing our data', default="localhost:9092")
    parser.add_argument('-tl','--listenTopic', help='Kafka Topic to Listen to', default="messages")
    parser.add_argument('-tw','--writeTopic', help='Kafka Topic to Write to', default="apache")
    parser.add_argument('-ch','--checkPoint', help='Path to Spark CheckPoint Directory', default="/tmp/checkpoint")
    parser.add_argument('-v','--verbose', help='Verbose', action='store_true')
    args = parser.parse_args()
    
    stock = StockPrediction()   
    stock.kafka_setup(args.kafkaListenServer, args.listenTopic)  
    
    
if __name__ == "__main__":
    main()        