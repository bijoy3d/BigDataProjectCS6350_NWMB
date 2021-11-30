# Main High level application function that reads streamed kafka story 
# trains the model and send the prediction to kafka
import argparse
from StockPred import StockPred
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json
from threading import Thread
from pyspark.sql.types import StringType, StructType, DoubleType
import time 
import shutil
from pyspark.sql import SQLContext
from sklearn.preprocessing import MinMaxScaler
from kafka import KafkaProducer
from timeit import default_timer as timer
import json 


class StockPrediction():
    # Init all variables for our application
    def __init__(self, kafkaWriteServer, datasetPath, modelPath):      
        self.spark = SparkSession.builder.appName("StockPrediction").config("spark.executor.memory", "70g").config("spark.driver.memory", "50g").config("spark.memory.offHeap.enabled",True).config("spark.memory.offHeap.size","16g").config("es.index.auto.create", "true").config('spark.sql.crossJoin.enabled',True).getOrCreate()
        self.spark.sparkContext.setLogLevel("ERROR")
        self.spark.conf.set("spark.sql.streaming.checkpointLocation", ".")
        self.spark.sql("set spark.sql.caseSensitive=true")
        self.sc = self.spark.sparkContext;
        self.sqlContext = SQLContext(self.sc)	
        self.datasetPath = datasetPath
        self.df = self.spark.read.option("inferSchema", "true").option("header", "true").csv(self.datasetPath)   
        self.lastdate = self.df.orderBy('Date',ascending=False).take(1)[0][0]
        self.predictedPrice = 0
        self.prevPredictedPrice = 0
        self.modelPath = modelPath
        self.pdf = self.df.toPandas()        
        self.opscaler = MinMaxScaler()
        self.ipscaler = MinMaxScaler() 
        
        # Serialized Json object 
        def serializer(message):
            return json.dumps(message).encode('utf-8')

        # Kafka Producer
        self.producer = KafkaProducer(bootstrap_servers=[kafkaWriteServer],value_serializer=serializer)     

    # This topic would be parsed by logstash and Kibana to show the analysis.
    def kafka_setup(self, kafkaListenServer, listenTopic, writeTopic):
        
        # Procees the newly received row batch here
        def process_row_batch(row, epoch): 
            # Processing needed only if we actually receive data. Skip otherwise
            if row.count() > 0:    
                print("New price bar from market : ", row.show())
                # Read the dataset from the storage
                newDF = self.spark.read.option("inferSchema", "true").option("header", "true").csv(self.datasetPath) 
                # Add the new data to the old dataset
                newDF = newDF.union(row) 
                mainDF = newDF.toPandas()
                # Delete the first row from the old dataset to avoid increasing the dataset size
                mainDF = mainDF.iloc[1: , :]
                # Save the dataset back to storage
                mainDF.to_csv(self.datasetPath, index=False)        
           
           
        # The Close Price is read here from the Kafka topic to which the alpaca json was sent by the alpaca API
        schema = StructType().add("Date", StringType()).add("Open", DoubleType()).add("Low", DoubleType()).add("High", DoubleType()).add("Close", DoubleType()).add("Volume", DoubleType()).add("vwap", DoubleType()).add("Trade_count", DoubleType())
        # Create DataSet representing the stream of input lines from kafka

        # Read the kafka story here. Waiting for the streaming data using spark streaming       
        lines = self.spark.readStream.format("kafka").option("kafka.bootstrap.servers", kafkaListenServer).option('subscribe', listenTopic).load().select(from_json(col("value").cast("string"), schema).alias("pdata"))      
        # Define the schema of the incoming data
        lines = lines.select(col("pdata.Date").alias('Date'),col("pdata.Open").alias('Open'),col("pdata.High").alias('High'),col("pdata.Low").alias('Low'),col("pdata.Close").alias('Close'),col("pdata.Volume").alias('Volume'),col("pdata.Trade_count").alias('Trade_count'),col("pdata.vwap").alias('vwap'))    
        # Write the incoming data to memory and call the process_row_batch for each new dataframe that arrives.
        lines = lines.writeStream.format("memory").foreachBatch(process_row_batch).option("checkpointLocation", "tmp").outputMode("append").start()         
        
        # As the received data frame is processed in the process_row_batch function we will continue with the LSTM 
        # model load, train and prediction here in the loop.
        while True :
            try:
                # Read the dataset
                mainDF = self.spark.read.option("inferSchema", "true").option("header", "true").csv(self.datasetPath)
                # Verify the datein the last row
                lastRow = mainDF.orderBy('Date',ascending=False).take(1)[0]
                # Get the new date
                newdate = lastRow[0]
                # Get the last closing price
                closePrice = lastRow[4]           
                        
                # Enter only if the data is new. Otherwise skip
                if newdate != self.lastdate : 
                    # Update last date with the new date
                    self.lastdate = newdate  
                
                    print("New price set from market. Start Training and Prediction Module")

                    start = timer()
                    # Create Stocks prediction object. Send it the dataset and set isGPU if needed                 
                    stocks = StockPred(mainDF.toPandas(),batch_size=72, isGPU=1)
                    # load saved model here
                    stocks.loadModel(self.modelPath)
                    # Train the model with the newly received data from the market
                    stocks.train_data(epoch=4)
                
                    # Pickle and save the model 
                    stocks.saveModel(self.modelPath)
                    
                    # Predict the next price using the newly saved model
                    self.predictedPrice = stocks.predict()
                    print("PREDICTED PRICE IS : ",self.predictedPrice)
                    print("Training took :", timer()-start) 
                
                    # Send the predicted price and actual price to Kafka
                    body = {
                        "currentPrice": float(closePrice),
                        "predictedPrice": float(self.prevPredictedPrice)
                    }

                    # Start sending the data from the next prediction onwards
                    if(self.prevPredictedPrice != 0):
                        self.producer.send(writeTopic, body)
                        self.producer.flush()
                
                    # Set the previous predicted price parameter
                    self.prevPredictedPrice = self.predictedPrice
            except Exception as e:
                print(e)
                print("Something got messed up. Price will be predicted for the next time step now")
                
            time.sleep(1)                       
         
        lines.awaitTermination()        
        
        
def main():     
    parser = argparse.ArgumentParser(description="Run Application to see Predicted Price for the stocks")
    parser.add_argument('-m','--modelPath', help='Path on local drive where to the save trained Model', default='C:\\Users\\bijpraka\\OneDrive - The University of Texas at Dallas\\Semester 4\\Big Data Management and Analytics\\Project\\stocksgithub\\BigDataProjectCS6350_NWMB\\stocks\\trainedModel')
    parser.add_argument('-b','--datasetPath', help='Path to the 1 week dataset', default='C:\\Users\\bijpraka\\OneDrive - The University of Texas at Dallas\\Semester 4\\Big Data Management and Analytics\\Project\\stocksgithub\\BigDataProjectCS6350_NWMB\\dataset\\apple_data.csv')
    parser.add_argument('-kl','--kafkaListenServer', help='Kafka Server:PORT from where we are listening for data', default="10.195.31.218:9092")
    parser.add_argument('-kw','--kafkaWriteServer', help='Kafka Server:PORT to where we are writing our data', default="10.195.31.218:9092")
    parser.add_argument('-tl','--listenTopic', help='Kafka Topic to Listen to', default="messages")
    parser.add_argument('-tw','--writeTopic', help='Kafka Topic to Write to', default="trades")
    parser.add_argument('-ch','--checkPoint', help='Path to Spark CheckPoint Directory', default="tmp")
    parser.add_argument('-v','--verbose', help='Verbose', action='store_true')
    args = parser.parse_args()
    
    stock = StockPrediction(args.kafkaWriteServer, args.datasetPath, args.modelPath, args.kafkaWriteServer)
    stock.kafka_setup(args.kafkaListenServer, args.listenTopic, args.writeTopic)  
    
    
if __name__ == "__main__":
    main()        
