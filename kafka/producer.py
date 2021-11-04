import time 
import json 
import random 
from datetime import datetime
from data_generator import generate_message
from kafka import KafkaProducer
import alpaca_trade_api as tradeapi
from alpaca_trade_api.stream import Stream
from alpaca_trade_api.common import URL
import threading
import websocket

# Messages will be serialized as JSON 
def serializer(message):
    return json.dumps(message).encode('utf-8')


# Kafka Producer
producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
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
        listen_message = {"action": "subscribe", "trades": ["CSCO"]}
        #{"action": "subscribe", "trades": ["AAPL"], "quotes": ["AMD", "CLDR"], "bars": ["*"]}
        #{"action": "subscribe", "quotes": ["CSCO"], "bars": ["*"]}
        ws.send(json.dumps(listen_message))


    def on_message(ws, message):
        #print("received a message")
        #print(message)
        allowed=['t']
        print(f'Sending @ {datetime.now()} | Message = {str(message)[1:-1]}')
        msgjson = json.loads(message[1:-1])
        if(msgjson['T'] in allowed):
            producer.send('messages', msgjson)
        
        # Sleep for a random number of seconds
        time_to_sleep = random.randint(1, 11)
        time.sleep(time_to_sleep)

    def on_close(ws, close_status, closemessage):
        print("closed connection ", closemessage)
            # print('disconnected from server')
        print ("Retry : %s" % time.ctime())
        time.sleep(1)
        connect_websocket() # retry per 10 seconds

    def on_error(ws, message):
        print("ERROR connection ", message)

    def connect_websocket():
        socket = "wss://stream.data.alpaca.markets/v2/iex"
        ws = websocket.WebSocketApp(socket, on_open=on_open, on_message=on_message, on_close=on_close, on_error=on_error)
        #websocket.enableTrace(True)
        ws.run_forever()

    connect_websocket()

def aftermarket():
    while(1):
        with open('dummy') as f:
            lines = f.readlines()
            for line in lines:
                print(json.loads(line))
                producer.send('messages', json.loads(line))
                # Sleep for a random number of seconds
                time_to_sleep = random.randint(1, 10)
                time.sleep(time_to_sleep)


if __name__ == '__main__':
#    print('AAPL moved {}% over the last 5 days'.format(percent_change))

    #contProd()
    #IF MARKET OPEN - CALL API TRUE
# Check if the market is open now.
    api = tradeapi.REST(api_key, api_secret, base_url, api_version='v2')
    clock = api.get_clock()
    if clock.is_open:
        alpaca()
    else:
        aftermarket()

    #print('The market is {}'.format('open.'  else 'closed.'))

    #alpaca()
    #ELSE - USE HISTORICAL
    #