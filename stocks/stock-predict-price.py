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
         
        self.spark = SparkSession.builder.appName("StockPrediction").config("spark.executor.memory", "70g").config("spark.driver.memory", "50g").config("spark.memory.offHeap.enabled",True).config("spark.memory.offHeap.size","16g").config("es.index.auto.create", "true").config('spark.sql.crossJoin.enabled',True).getOrCreate()
        self.spark.sparkContext.setLogLevel("ERROR")
        self.spark.conf.set("spark.sql.streaming.checkpointLocation", "/tmp/checkpoint")
        self.spark.sql("set spark.sql.caseSensitive=true")
        self.sc = self.spark.sparkContext;
        self.sqlContext = SQLContext(self.sc)	
        #self.df = self.spark.read.option("inferSchema", "true").option("header", "true").csv("/home/nithyashanmugam/TradeApp/apple_5min_data_1week.csv")
        self.df = self.spark.read.option("inferSchema", "true").option("header", "true").csv("/home/nithyashanmugam/TradeApp/stream.csv")
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
        
        def process_row_batch(row, epoch):
            print("inside process row batch")            
            print(row.show())
            newDF = self.spark.read.option("inferSchema", "true").option("header", "true").csv('/home/nithyashanmugam/TradeApp/stream.csv' ) 
            newDF = newDF.union(row) 
            mainDF = newDF.toPandas()
            mainDF = mainDF.iloc[1: , :]
            mainDF.to_csv('/home/nithyashanmugam/TradeApp/stream.csv', index=False) 
            print(mainDF.tail(5))            
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
           mainNewDF = mainNewDF.iloc[1: , :]
           mainNewDF.to_csv('/home/nithyashanmugam/TradeApp/stream.csv', index=False)
           print(mainNewDF.tail(4))
           
           opscaler = MinMaxScaler()
           ipscaler = MinMaxScaler()
           inputs=mainNewDF          
           targets = inputs.filter(["Open"], axis=1)
           targets.columns = ['target']
           targets["target"]=targets['target'][1:].reset_index(drop=True)
           targets.iloc[-1]['target'] = targets.iloc[:-1]['target'].mean()
           inputs[['Open','High','Low','Close','Volume','Trade_count','vwap']] = ipscaler.fit_transform(inputs[['Open','High','Low','Close','Volume','Trade_count','vwap']])
           targets[['target']] = opscaler.fit_transform(targets[['target']])           
           #lstm = LSTM.LSTM(train_data=inputs, targets=targets, batch_size=288, debug=0, test=0)
           #lstm.train(epoch=1, lr=1)

          
           #load saved trained model
           #train the model with new batch
           #save the new model
           #predict price for next time step and print
           #send it to kafka - actual price, predicted price
           
        
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
        #lines = lines.writeStream.format("memory").foreach(process_row).option("checkpointLocation", "/tmp/checkpoint/").outputMode("append").start() 
             
        #mainDF = mainDF.join(lines) 
        #lines = lines.writeStream.format("console").foreach(process_row).outputMode("append").start() 
        lines = lines.writeStream.format("memory").foreachBatch(process_row_batch).option("checkpointLocation", "/tmp/checkpoint/").outputMode("append").start()         
        
        while True :
            mainDF = self.spark.read.csv('/home/nithyashanmugam/TradeApp/stream.csv')   
            #mainNewDF = mainDF.toPandas()
            print("inside while")
            print(mainDF)
            time.sleep(5)
             
            
        # header  = mainDF.first()[0]
        # mainDF.filter(col("value").contains(header))
        # mainDF = mainDF.iloc[1: , :]        
        # #mainDF.write.csv('/home/nithyashanmugam/TradeApp/stream.csv')   
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