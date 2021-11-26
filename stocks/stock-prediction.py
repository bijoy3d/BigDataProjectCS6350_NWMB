import argparse
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import HashingTF, IndexToString, StringIndexer, Tokenizer
from pyspark.ml.pipeline import Pipeline
from pyspark.ml.tuning import CrossValidator, CrossValidatorModel, ParamGridBuilder
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json
from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.sql.types import StringType, StructType
from keras.models import Sequential
from keras.layers import LSTM, Dense
import sys
sys.path.append("..")
from lstm import LSTM 

spark = SparkSession.builder.appName("StockPrediction").config("spark.executor.memory", "70g").config("spark.driver.memory", "50g").config("spark.memory.offHeap.enabled",True).config("spark.memory.offHeap.size","16g").config("es.index.auto.create", "true").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# Train and build the model if not available already
def build_model(modelFile, inputFile):
    df = spark.read.option("inferSchema", "true").option("header", "true").csv(inputFile)
    
    close_data = df['Close'].values
    close_data = close_data.reshape((-1,1))
    
    split_percent = 0.80
    split = int(split_percent*len(close_data))

    close_train = close_data[:split]
    close_test = close_data[split:]
    
    look_back = 1

    train_generator = TimeseriesGenerator(close_train, close_train, length=look_back, batch_size=20) 

    cvModel = Sequential()
    cvModel.add(
        LSTM(10,
            activation='relu',
            input_shape=(look_back,1))
    )
    cvModel.add(Dense(1))
    cvModel.compile(optimizer='adam', loss='mse')

    num_epochs = 5
    cvModel.fit_generator(train_generator, epochs=num_epochs, verbose=1)

    # Save the trained model to disk.
    cvModel.write().overwrite().save(modelFile)

# Load the model from the local file system to save on time. 
def loadModel(modelFile):
    return CrossValidatorModel.read().load(modelFile)

# Find the sentiment from the trained model and write the sentiment to Kafka topic
# This topic would be parsed by logstash and Kibana to show the analysis.
def kafka_setup(trainedModel, kafkaListenServer, kafkaWriteServer, listenTopic, writeTopic):
    print(writeTopic)
    
    # This function would send the Close Price to the trained model to find the predicted price.
    def prdeictStockPrice(cvModel):        
        prediction = trainedModel.transform(cvModel)
        converter = IndexToString(inputCol="prediction", outputCol="PredictedPrice", labels=labels)
        converted = converter.transform(prediction)
        selected = converted.select("Close", "PredictedPrice").withColumnRenamed("PredictedPrice","value")
        selected.writeStream.format("kafka").option("kafka.bootstrap.servers", kafkaWriteServer).option('topic', writeTopic).start()
        spark.streams.awaitAnyTermination()

    # The tweet text is read here from the Kafka topic to which the tweet json was sent by the twitterConnect API

    schema = StructType().add("Date", StringType()).add("Open", StringType()).add("High", StringType()).add("Close", StringType()).add("Volume", StringType()).add("vwap", StringType())
    # Create DataSet representing the stream of input lines from kafka
    lines = spark.readStream.format("kafka").option("kafka.bootstrap.servers", kafkaListenServer).option('subscribe', listenTopic).load().select(from_json(col("value").cast("string"), schema).alias("pdata"))
    closePrice = lines.select(col("pdata.Close").alias('Close'))
    prdeictStockPrice(closePrice)

def main():
    parser = argparse.ArgumentParser(description="Run Application to see Tweet Sentiments")
    parser.add_argument('-m','--modelFile', help='Path on local drive where to the save trained Model if -b is enabled or just load the trained model from', default='/home/bijpraka/spark/assg3/nnmodel')
    parser.add_argument('-b','--buildModel', help='Option to specify to build the model and path on local drive to CSV File. If not enabled then use the trained model stored in -m', action='store_true')
    parser.add_argument('-c','--csvFile', help='Path to CSV File where training data is saved. This would be used if -b is enabled', default="/home/bijpraka/spark/assg3/Tweets.csv")
    parser.add_argument('-kl','--kafkaListenServer', help='Kafka Server:PORT from where we are listening for data', default="localhost:9092")
    parser.add_argument('-kw','--kafkaWriteServer', help='Kafka Server:PORT to where we are writing our data', default="localhost:9092")
    parser.add_argument('-tl','--listenTopic', help='Kafka Topic to Listen to', default="messages")
    parser.add_argument('-tw','--writeTopic', help='Kafka Topic to Write to', default="apache")
    parser.add_argument('-ch','--checkPoint', help='Path to Spark CheckPoint Directory', default="/tmp/checkpoint")
    parser.add_argument('-v','--verbose', help='Verbose', action='store_true')
    args = parser.parse_args()
    spark.conf.set("spark.sql.streaming.checkpointLocation", args.checkPoint)

    # Build the model if not present already
    if(args.buildModel):
        build_model(args.modelFile, args.csvFile)

    kafka_setup(loadModel(args.modelFile), args.kafkaListenServer, args.kafkaWriteServer, args.listenTopic, args.writeTopic)
    
if __name__ == "__main__":
    main()
