from .LSTM import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

class StockPred():
    def __init__(self, inputs, batch_size=144):
        self.opscaler = MinMaxScaler()
        self.ipscaler = MinMaxScaler()
        self.inputs=inputs
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
        self.lstm = LSTM(train_data=self.inputs, targets=self.targets, batch_size=batch_size, debug=0, test=0)
        #self.split_data()
    
    def scale_data(self):
        self.inputs[['Open','High','Low','Close','Volume','Trade_count','vwap']] = self.ipscaler.fit_transform(self.inputs[['Open','High','Low','Close','Volume','Trade_count','vwap']])
        self.targets[['target']] = self.opscaler.fit_transform(self.targets[['target']])
    
    def split_data(self):
        self.iptrain, self.iptest, self.optrain, self.optest = train_test_split(self.inputs, self.targets, test_size=0.2, shuffle=False)
        
    def train_data(self, epoch=3, lr=1):
        self.lstm.train(epoch=epoch, lr=lr)
        
    def predict(self):
        return self.lstm.goPredict(self.inputs, opscaler=self.opscaler, ipscaler=self.ipscaler)

    def loadModel(self, filepath):
        self.lstm.loadModel(filepath)

    def saveModel(self, filepath):
        self.lstm.saveModel(filepath)

    def validate(self):
        self.lstm.goValidate(self.iptest, self.optest, self.opscaler, self.ipscaler)