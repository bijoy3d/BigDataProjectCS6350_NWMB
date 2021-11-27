from .LSTM import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

class StockPred():
    def __init__(self, inputs):
        self.opscaler = MinMaxScaler()
        self.ipscaler = MinMaxScaler()
        self.inputs=inputs
        self.inputs.drop("Date", axis=1, inplace=True)
        self.targets = self.inputs.filter(["Open"], axis=1)
        self.targets.columns = ['target']
        self.targets["target"]=self.targets['target'][1:].reset_index(drop=True)
        self.targets.iloc[-1]['target'] = self.targets.iloc[:-1]['target'].mean()
        self.scale_data()
        self.split_data()
    
    def scale_data(self):
        self.inputs[['Open','High','Low','Close','Volume','Trade_count','vwap']] = self.ipscaler.fit_transform(self.inputs[['Open','High','Low','Close','Volume','Trade_count','vwap']])
        self.targets[['target']] = self.opscaler.fit_transform(self.targets[['target']])
    
    def split_data(self):
        self.iptrain, self.iptest, self.optrain, self.optest = train_test_split(self.inputs, self.targets, test_size=0.2, shuffle=False)
        
    def train_data(self, epoch=3, lr=1, batch_size=288):
        self.lstm = LSTM(train_data=self.iptrain, targets=self.optrain, batch_size=batch_size, debug=0, test=0)
        self.lstm.train(epoch=epoch, lr=1)
        
    def validate(self):
        self.lstm.goValidate(self.iptest, self.optest, self.opscaler, self.ipscaler)