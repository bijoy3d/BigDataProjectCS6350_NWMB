# Helper class for Stocks to access the LSTM module
from lstm.LSTM import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from lstm.lstm_gpu import *
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning, NumbaWarning
import warnings
import logging;

class StockPred():
    # Init the LSTM module and all data needed for it. We decide whether it will be GPU mode or not here
    def __init__(self, inputs, batch_size=144, isGPU=0):
        self.opscaler = MinMaxScaler()
        self.ipscaler = MinMaxScaler()
        self.isGPU = isGPU
        self.inputs=inputs
        self.batch_size = batch_size
        self.inputs.drop("Date", axis=1, inplace=True)
        self.targets = self.inputs.filter(["Close"], axis=1)
        self.targets.columns = ['target']
        self.targets["target"]=self.targets['target'][1:].reset_index(drop=True)
        try:
            print("target last row value ", self.targets.iloc[-1]['target'])
            self.targets.iloc[-1]['target'] = self.targets.iloc[:-1]['target'].mean()
            print("target last row value ", self.targets.iloc[-1]['target'])
        except:
            print("Some exception. Ignoring as it is not fatal")
        self.scale_data()
        if isGPU:
            warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
            warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
            warnings.simplefilter('ignore', category=NumbaWarning)
            logger = logging.getLogger("numba");
            logger.setLevel(logging.ERROR)
            self.p = init(train_data=self.inputs, targets=self.targets, batch_size=batch_size, debug=0, test=0)
        else:
            self.lstm = LSTM(train_data=self.inputs, targets=self.targets, batch_size=batch_size, debug=0, test=0)
    
    # Data scaling functiom
    def scale_data(self):
        self.inputs[['Open','High','Low','Close','Volume','Trade_count','vwap']] = self.ipscaler.fit_transform(self.inputs[['Open','High','Low','Close','Volume','Trade_count','vwap']])
        self.targets[['target']] = self.opscaler.fit_transform(self.targets[['target']])
    
    # Data splitter if needed
    def split_data(self):
        self.iptrain, self.iptest, self.optrain, self.optest = train_test_split(self.inputs, self.targets, test_size=0.2, shuffle=False)
        
    # Call LSTM trainer
    def train_data(self, epoch=3, lr=1):
        if self.isGPU:
            self.p=train(self.p, self.batch_size, self.targets, self.inputs, epoch=epoch, lr=lr)
        else:
            self.lstm.train(epoch=epoch, lr=lr)
        
    # Call LSTM predict function
    def predict(self):
        if self.isGPU:
            return goPredict(self.p, self.targets, self.inputs, self.batch_size, opscaler=self.opscaler, ipscaler=self.ipscaler)    
        else:
            return self.lstm.goPredict(self.inputs, opscaler=self.opscaler, ipscaler=self.ipscaler)

    # Load pickled LSTM model
    def loadModel(self, filepath):
        if self.isGPU:
            self.p=loadModel(filepath)
        else:
            self.lstm.loadModel(filepath)

    # Pickle and save LSTM model
    def saveModel(self, filepath):
        if (self.isGPU):
            saveModel(filepath, self.p)
        else:
            self.lstm.saveModel(filepath)

    # Validation function if needed
    def validate(self):
        self.lstm.goValidate(self.iptest, self.optest, self.opscaler, self.ipscaler)