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
import lstm.LSTM as LSTM
from lstm.StockPred import StockPred
from pyspark.sql import SQLContext
from sklearn.preprocessing import MinMaxScaler
from pyspark.sql import Row
from kafka import KafkaProducer
from timeit import default_timer as timer
import json 


class StockPrediction():
    def __init__(self):      
        dir_path = '/tmp/checkpoint'
        try:
            shutil.rmtree(dir_path)
        except OSError as e:
            print("Error: %s : %s" % (dir_path, e.strerror))
         
        self.spark = SparkSession.builder.appName("StockPrediction").config("spark.executor.memory", "70g").config("spark.driver.memory", "50g").config("spark.memory.offHeap.enabled",True).config("spark.memory.offHeap.size","16g").config("es.index.auto.create", "true").config('spark.sql.crossJoin.enabled',True).getOrCreate()
        self.spark.sparkContext.setLogLevel("ERROR")
        self.spark.conf.set("spark.sql.streaming.checkpointLocation", "/tmp/checkpoint")
        self.spark.sql("set spark.sql.caseSensitive=true")
        self.sc = self.spark.sparkContext;
        self.sqlContext = SQLContext(self.sc)	
        self.df = self.spark.read.option("inferSchema", "true").option("header", "true").csv("/home/nithyashanmugam/TradeApp/stream.csv")   
        self.lastdate = self.df.orderBy('Date',ascending=False).take(1)[0][0]
        self.predictedPrice = 0
        self.modelPath = "/home/nithyashanmugam/TradeApp/trainedModel"
        print(self.lastdate)  
        self.pdf = self.df.toPandas()        
        self.opscaler = MinMaxScaler()
        self.ipscaler = MinMaxScaler() 
        
        # Messages will be serialized as JSON 
        def serializer(message):
            return json.dumps(message).encode('utf-8')
        # Kafka Producer
        self.producer = KafkaProducer(bootstrap_servers=['localhost:9092'],value_serializer=serializer)     

    # This topic would be parsed by logstash and Kibana to show the analysis.
    def kafka_setup(self,kafkaListenServer, listenTopic):
        
        def process_row_batch(row, epoch):              
            if row.count() > 0:    
                print("inside row count", row.show())
                newDF = self.spark.read.option("inferSchema", "true").option("header", "true").csv('/home/nithyashanmugam/TradeApp/stream.csv' ) 
                newDF = newDF.union(row) 
                mainDF = newDF.toPandas()
                mainDF = mainDF.iloc[1: , :]
                mainDF.to_csv('/home/nithyashanmugam/TradeApp/stream.csv', index=False)        
           
           
        # The Close Price is read here from the Kafka topic to which the alpaca json was sent by the alpaca API
        #dataframe -> csv 
        schema = StructType().add("Date", StringType()).add("Open", StringType()).add("Low", StringType()).add("High", StringType()).add("Close", StringType()).add("Volume", StringType()).add("vwap", StringType()).add("Trade_count", StringType())
        # Create DataSet representing the stream of input lines from kafka
       
        lines = self.spark.readStream.format("kafka").option("kafka.bootstrap.servers", kafkaListenServer).option('subscribe', listenTopic).load().select(from_json(col("value").cast("string"), schema).alias("pdata"))      
        lines = lines.select(col("pdata.Date").alias('Date'),col("pdata.Open").alias('Open'),col("pdata.High").alias('High'),col("pdata.Low").alias('Low'),col("pdata.Close").alias('Close'),col("pdata.Volume").alias('Volume'),col("pdata.Trade_count").alias('Trade_count'),col("pdata.vwap").alias('vwap'))    
        
        lines = lines.writeStream.format("memory").foreachBatch(process_row_batch).option("checkpointLocation", "/tmp/checkpoint/").outputMode("append").start()         
        
        while True :
            mainDF = self.spark.read.option("inferSchema", "true").option("header", "true").csv('/home/nithyashanmugam/TradeApp/stream.csv')
            newdate = mainDF.orderBy('Date',ascending=False).take(1)[0][0]
            closePrice = mainDF.orderBy('Date',ascending=False).take(1)[0][4]
           
            print(newdate,self.lastdate)            
           
            if newdate != self.lastdate : 
                self.lastdate = newdate  
                
                print("we got new record")
                start = timer()                 
                stocks = StockPred(mainDF.toPandas(),batch_size=72)
                #load saved model here
                stocks.loadModel(self.modelPath)
                #stocks.train_data(epoch=4)
                
                #save the model 
                stocks.saveModel(self.modelPath)
                  
                self.predictedPrice = stocks.predict();
                print("PREDICTION DONE",self.predictedPrice)
                print("Training took :", timer()-start) 
              
                #predict the price and send to kafka
                body = {
                    "currentPrice": float(closePrice),
                    "predictedPrice": float(self.predictedPrice)
                }
                self.producer.send('apache', body)
                self.producer.flush()
                
            time.sleep(5)            
            
         
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