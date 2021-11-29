#!/usr/bin/env python
# coding: utf-8

import time 
import json 
import random 
from datetime import datetime
from kafka import KafkaProducer
import alpaca_trade_api as tradeapi
from alpaca_trade_api.stream import Stream
from alpaca_trade_api.common import URL
import threading
import websocket
import pandas as pd

data = pd.read_json("C:\\Users\\bijpraka\\OneDrive - The University of Texas at Dallas\\Semester 4\\Big Data Management and Analytics\\Project\\stocksgithub\\BigDataProjectCS6350_NWMB\\dataset\\dummy_new") 
print(data)

# Messages will be serialized as JSON 
def serializer(message):
    return json.dumps(message).encode('utf-8')


# Kafka Producer
producer = KafkaProducer(
    bootstrap_servers=['10.195.31.218:9092'],
    value_serializer=serializer
)

data_url = 'wss://data.alpaca.markets'
base_url = 'https://paper-api.alpaca.markets'
api_key= 'PKZEXWB0J5CEU2JJZXV8'
api_secret = 'kPy7mGsSnzeFnTgpMKFAQG58blbwqYgPiTeEeUPo'

def alpaca():
    def on_open(ws):
        print("opened")
        auth_data = {"action": "auth", "key": api_key, "secret": api_secret}
        print(auth_data)
        ws.send(json.dumps(auth_data))
        listen_message = {"action": "subscribe", "bars": ["AAPL"]}       
        ws.send(json.dumps(listen_message))


    def on_message(ws, message):        
        allowed=['b']
        msgjson = json.loads(message[1:-1])
        if(msgjson['T'] in allowed):  
            print(msgjson)
            body = {
                "Date": str(msgjson["t"]),
                "Open": str(msgjson["o"]),             
                "High": str(msgjson["h"]),
                "Low": str(msgjson["l"]),
                "Close": str(msgjson["c"]),    
                "Volume": str(msgjson["v"]),
                "Trade_count": str(msgjson["n"]),
                "vwap": str(msgjson["vw"])
            }            
            producer.send('messages', body)
        
        # Get data every 5 mins
        time.sleep(300)

    def on_close(ws, close_status, closemessage):
        print("closed connection ", closemessage)        
        print ("Retry : %s" % time.ctime())
        time.sleep(1)
        connect_websocket() # retry per 10 seconds

    def on_error(ws, message):
        print("ERROR connection ", message)

    def connect_websocket():
        socket = "wss://stream.data.alpaca.markets/v2/iex"
        ws = websocket.WebSocketApp(socket, on_open=on_open, on_message=on_message, on_close=on_close, on_error=on_error)
        ws.run_forever()

    connect_websocket()

def aftermarket():   
    for index, row in data.iterrows():  
        producer.send('messages', json.loads(row.to_json(date_format = "iso")))
        time.sleep(300)

if __name__ == '__main__':

    api = tradeapi.REST(api_key, api_secret, base_url, api_version='v2')
    clock = api.get_clock()
    if clock.is_open:
        alpaca()
    else:
        aftermarket()

    