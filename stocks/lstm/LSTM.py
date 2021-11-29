#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from io import StringIO

import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def sigmoid(x):
    """
    Computes the element-wise sigmoid activation function for an array x.

    Args:
     `x`: the array where the function is applied
     `derivative`: if set to True will return the derivative instead of the forward pass
    """
    x_safe = x + 1e-12
    return 1 / (1 + np.exp(-x_safe))

class LSTM():
    def __init__(self, train_data, targets, batch_size=2, debug=1, test=1):
        #4 gates
        # input_activation = tanh(wa [inner] input + ua [inner] prev_output + ba)
        # input_gate  = sigmoid(wi [inner] input + ui [inner] prev_output + bi)
        # forget_gate = sigmoid(wf [inner] input + uf [inner] prev_output + bf)
        # output_gate = sigmoid(wo [inner] input + uo [inner] prev_output + bo)

        # 2 states
        # internal_state = (input_activation [Element wise] input_gate) + (forget_gate [Element wise] prev_internal_state)
        # output = tanh(internal_state) [Element wise] output_gate

        if batch_size >= len(train_data):
            print(f'Batch Size {batch_size} should be less than the size of the dataset {len(train_data)}')
            return None
        
        # To enable test mode. Batch size would be set as 2
        self.test = test
        
        # The number of records that would go inside the LSTM at one time. A sequence of records.
        self.batch_size = batch_size
        numFeats = train_data.shape[1] ###### CHANGE IT TO GET DYNAMICALLY FROM INPUT
        # Enable debug logs
        self.debug = debug
        
        # Training Data
        self.train_data = train_data
        # Target of training data
        self.targets = targets
        
        
        # input_activation
        self.wa = np.random.random((numFeats, 1))
        self.ua = np.random.random((1, 1))
        self.ba = np.random.random((1, 1))

        # input_gate
        self.wi = np.random.random((numFeats, 1))
        self.ui = np.random.random((1, 1))
        self.bi = np.random.random((1, 1))

        # forget_gate
        self.wf = np.random.random((numFeats, 1))
        self.uf = np.random.random((1, 1))
        self.bf = np.random.random((1, 1))

        # output_gate
        self.wo = np.random.random((numFeats, 1))
        self.uo = np.random.random((1, 1))
        self.bo = np.random.random((1, 1))

        # Clean and init LSTM
        self.cleanLSTM()


        if self.test:
            self.batchSize = 1
            self.wa[0]=0.45
            self.wa[1]=0.25
            self.ua[0]=0.15
            self.ba[0]=0.2
            self.wi[0]=0.95
            self.wi[1]=0.8
            self.ui[0]=0.8
            self.bi[0]=0.65
            self.wf[0]=0.7
            self.wf[1]=0.45
            self.uf[0]=0.1
            self.bf[0]=0.15
            self.wo[0]=0.6
            self.wo[1]=0.4
            self.uo[0]=0.25
            self.bo[0]=0.1
        
    def cleanLSTM(self):
        # Forward Propogation Parameters
        self.prev_input_activation = 0
        self.prev_input_gate = 0
        self.prev_forget_gate = 0
        self.prev_output_gate = 0
        
        self.input_activation = 0
        self.input_gate = 0
        self.forget_gate = 0
        self.output_gate = 0
        self.internal_state = np.zeros((1, 1))
        self.output = np.zeros((1, 1))

        self.prev_input_activations = []
        self.prev_input_gates  = []
        self.prev_output_gates  = []
        self.prev_forget_gates  = []
        self.prev_internal_states  = []
        self.prev_outputs = []
        
        
        # Backward Propogation Parameters
        self.stacked_ip_weights = []
        self.stacked_op_weights = []
        
        self.der_internal_state_future = np.zeros((1, 1))
        self.delta_op_future = np.zeros((1, 1))
        
        self.input_weight_derivatives = 0
        self.output_weight_derivatives = 0
        self.bias_derivatives = 0

    def update_lstmData(self, lr=.01):
        wa, ua, ba, wi, ui, bi, wf, uf, bf, wo, uo, bo = self.getLSTMparms()
        dip = self.input_weight_derivatives
        dop = self.output_weight_derivatives
        db = self.bias_derivatives
        
        t = wa.T - (dip*lr)
        wa.T[0] = t[0].T
        t = wi.T - (dip*lr)
        wi.T[0] = t[1].T
        t = wf.T - (dip*lr)
        wf.T[0] = t[2].T
        t = wo.T - (dip*lr)
        wo.T[0] = t[3].T
        ua = ua.T - (dop*lr)[0]
        ui = ui.T - (dop*lr)[1]
        uf = uf.T - (dop*lr)[2]
        uo = uo.T - (dop*lr)[3]
        ba = ba.T - (db*lr)[0]
        bi = bi.T - (db*lr)[1]
        bf = bf.T - (db*lr)[2]
        bo = bo.T - (db*lr)[3]
        self.wa=wa
        self.ua=ua
        self.ba=ba
        self.wi=wi
        self.ui=ui
        self.bi=bi
        self.wf=wf
        self.uf=uf
        self.bf=bf
        self.wo=wo
        self.uo=uo
        self.bo=bo

    def printLSTMparms(self):
        lstmData = self.getLSTMparms()
        print('wa:', lstmData[0].shape)
        print('ua:', lstmData[1].shape)
        print('ba:', lstmData[2].shape)
        print('wi:', lstmData[3].shape)
        print('ui:', lstmData[4].shape)
        print('bi:', lstmData[5].shape)
        print('wf:', lstmData[6].shape)
        print('uf:', lstmData[7].shape)
        print('bf:', lstmData[8].shape)
        print('wo:', lstmData[9].shape)
        print('uo:', lstmData[10].shape)
        print('bo:', lstmData[11].shape)


        print('wa:', lstmData[0])
        print('ua:', lstmData[1])
        print('ba:', lstmData[2])
        print('wi:', lstmData[3])
        print('ui:', lstmData[4])
        print('bi:', lstmData[5])
        print('wf:', lstmData[6])
        print('uf:', lstmData[7])
        print('bf:', lstmData[8])
        print('wo:', lstmData[9])
        print('uo:', lstmData[10])
        print('bo:', lstmData[11])

    def lstm_data_transform(self, ip=None):
        """ Changes data to the format for LSTM training 
    for sliding window approach """
        # Prepare the list for the transformed data
        X, y = list(), list()
        if ip is not None:
            data = ip
        else:
            data = self.train_data
        # Loop of the entire data set
        for i in range(data.shape[0]):
            # compute a new (sliding window) index
            end_ix = i + self.batch_size

            # if index is larger than the size of the dataset, we stop
            if end_ix >= data.shape[0]:
                break
            # Get a sequence of data for x
            seq_X = data[i:end_ix]
            # Get only the last element of the sequency for y
            seq_y = self.targets[i:end_ix]
            # Append the list with sequencies
            X.append(seq_X)
            y.append(seq_y)
        # Make final arrays
        x_array = np.array(X)
        y_array = np.array(y)
        return x_array, y_array
    
    def plog(self, *msg, f=0):
        if self.debug or f:
            print(*msg)

    def setLSTMparms(self, parms):
        self.wa, self.ua, self.ba, self.wi, self.ui, self.bi, self.wf, self.uf, self.bf, self.wo, self.uo, self.bo = parms
        
    def getLSTMparms(self):
        return self.wa, self.ua, self.ba, self.wi, self.ui, self.bi, self.wf, self.uf, self.bf, self.wo, self.uo, self.bo

    def goForward(self, ipt, train=1):
        #4 gates
        # input_activation = tanh(wa [inner] input + ua [inner] prev_output + ba)
        # input_gate  = sigmoid(wi [inner] input + ui [inner] prev_output + bi)
        # forget_gate = sigmoid(wf [inner] input + uf [inner] prev_output + bf)
        # output_gate = sigmoid(wo [inner] input + uo [inner] prev_output + bo)

        # 2 states
        # internal_state = (input_activation [Element wise] input_gate) + (forget_gate [Element wise] prev_internal_state)
        # output = tanh(internal_state) [Element wise] output_gate
    
        plog = self.plog
        wa, ua, ba, wi, ui, bi, wf, uf, bf, wo, uo, bo = self.getLSTMparms()
        
        po = self.output
        ps = self.internal_state
        plog("wa : ", wa.T)
        plog("ipt : ", ipt)
        plog("ua : ", ua)
        plog("po : ", po)
        plog("ba : ", ba)

        plog("ipt T shape", ipt.T.shape)
        plog("po shape", po.shape)
            
           
        # input activation
        self.input_activation = np.tanh((np.inner(wa.T, ipt)) + (np.inner(ua, po)) + ba)
        ia = self.input_activation

        # input gate
        self.input_gate = sigmoid((np.inner(wi.T, ipt)) + (np.inner(ui, po)) + bi)

        # forget gate
        self.forget_gate = sigmoid((np.inner(wf.T, ipt)) + (np.inner(uf, po)) + bf)
        
        # output gate
        self.output_gate = sigmoid((np.inner(wo.T, ipt)) + (np.inner(uo, po)) + bo)

        # internal state
        self.internal_state = (np.multiply(ia, self.input_gate)) + (np.multiply(self.forget_gate, ps))

        # output
        self.output = np.multiply(np.tanh(self.internal_state), self.output_gate)
        
        if train:
            self.prev_input_activations.append(ia)
            self.prev_input_gates.append(self.input_gate)
            self.prev_forget_gates.append(self.forget_gate)
            self.prev_output_gates.append(self.output_gate)
            self.prev_internal_states.append(self.internal_state)
            self.prev_outputs.append(self.output)

        plog("input_activation = ",ia)
        plog("input gate : ", self.input_gate)
        plog("forget gate : ", self.forget_gate)
        plog("output gate : ",self.output_gate)
        plog("internal state", self.internal_state)
        plog("output = ",self.output)
        plog("----------------------------------")
        return self.output
        
    def stackWeights(self):
        stacked_ip_weights = np.copy(self.wa)
        stacked_ip_weights = np.column_stack((stacked_ip_weights, self.wi))
        stacked_ip_weights = np.column_stack((stacked_ip_weights, self.wf))
        stacked_ip_weights = np.column_stack((stacked_ip_weights, self.wo))
        self.stacked_ip_weights = stacked_ip_weights
        
        stacked_op_weights = np.copy(self.ua)
        stacked_op_weights = np.column_stack((stacked_op_weights, self.ui))
        stacked_op_weights = np.column_stack((stacked_op_weights, self.uf))
        stacked_op_weights = np.column_stack((stacked_op_weights, self.uo))
        self.stacked_op_weights = stacked_op_weights

    def travelBack(self, targets, inputs):

        plog = self.plog
        # Unpack parameters
        wa, ua, ba, wi, ui, bi, wf, uf, bf, wo, uo, bo = self.getLSTMparms()
        tempo = np.zeros((1, 1))
        loss=0
        plog("Targets is",targets)
        plog("Inputs is",inputs)

        for t in reversed(range(len(self.prev_outputs))):

            output = self.prev_outputs[t]
            target = targets[t]
            
            next_forget_gate = np.zeros((1, 1)) if (t==len(self.prev_outputs)-1) else self.prev_forget_gates[t+1]
            
            plog("previous outputs = ", str(self.prev_outputs))
            plog("target = ",str(target))
            plog("output = ", str(output))
            
            # Track loss
            loss = (np.power((target - output),2))/2
            plog("loss = ", str(loss), f=0)

            # derivative of loss with respect to output
            der_loss_wrt_output = output - target
            plog("der_loss_wrt_output = ", der_loss_wrt_output)

            # derivative of output
            der_output = der_loss_wrt_output + self.delta_op_future
            plog("der_output = ", der_output)

            # derivative of internal state
            pog = self.prev_output_gates[t]
            ps = self.prev_internal_states[t]
            dfis = der_output * pog * (1 - (np.tanh(ps))**2 ) + (self.der_internal_state_future * next_forget_gate)
            self.der_internal_state_future = dfis
            plog("der internal state = ", dfis)
            plog("pog : ", pog)
            plog("ps : ", ps)


            pig = self.prev_input_gates[t]
            pia = self.prev_input_activations[t]
            der_input_activation = dfis * pig * (1 - pia**2)
            plog("der_input_activation = ", der_input_activation)
            stacked_ders = np.copy(der_input_activation)

            der_inputg = dfis * pia * pig * (1 - pig)
            stacked_ders = np.row_stack((stacked_ders, der_inputg))
            plog("der_input = ", der_inputg)

            pps = tempo if t==0 else self.prev_internal_states[t-1] 
            pfg = self.prev_forget_gates[t]
            der_forgetg = dfis * pps * pfg * (1 - pfg)
            stacked_ders = np.row_stack((stacked_ders, der_forgetg))
            plog("der_forget = ", der_forgetg)   
            
            plog("pps : ", pps, t-1)
            plog("pfg : ", pfg)
            plog("dfis : ", str(dfis))

            der_outputg = der_output * np.tanh(ps) * pog * (1 - pog)
            stacked_ders = np.row_stack((stacked_ders, der_outputg))
            plog("der_output = ", der_outputg)

            self.stackWeights()


            der_input_state = np.dot(self.stacked_ip_weights, stacked_ders)
            plog("der_input_state = ", der_input_state)

            der_output_state = np.dot(self.stacked_op_weights, stacked_ders)
            plog("der_output_state = ", der_output_state)
            self.delta_op_future = der_output_state

            plog("inputs t is : ",str(t), np.array([inputs[0][t]]))
            der_input_weight = np.dot(stacked_ders, np.array([inputs[0][t]]))
            self.input_weight_derivatives += der_input_weight
            plog("der_input_weight : ", der_input_weight)

            po = tempo if t==0 else self.prev_outputs[t-1] 
            der_op_weight = np.dot(stacked_ders, po)
            self.output_weight_derivatives += der_op_weight
            plog("der_op_weight : ", der_op_weight)

            self.bias_derivatives += stacked_ders
        return loss
    
    def train(self, epoch=2, lr=.01):
        plog = self.plog
        ip_batches, op_batches = self.lstm_data_transform()
        #print(op_batches)
        count = 1
        for runit in range (epoch):
            print("Running EPOCH ", runit+1)

            for ipbatch,opbatch in zip(ip_batches, op_batches):
                self.cleanLSTM()
                if count % 100 ==0:
                    print("Round "+str(count))
                plog("Round "+str(count)," opbatch is : ", opbatch)
                for ip in ipbatch:
                    plog("Round "+str(count)," ip is ",ip)
                    self.goForward(np.array([ip]))
                loss = self.travelBack(opbatch, np.array([ipbatch]))
                plog("Round "+str(count)," Forward and Backward DONE", f=0)
                plog("Round "+str(count)," OP DONE")
                plog("Round "+str(count)," OLD WEIGHTS")
                #self.printLSTMparms()
                self.update_lstmData(lr)
                plog("Round "+str(count), " NEW WEIGHTS")
                #self.printLSTMparms()
                count+=1
           # if runit % 10 ==0:
            print("loss at epoch",runit,"is ", loss)
    
    def goPredict(self, inputs, opscaler=None, ipscaler=None):
        plog = self.plog
        ip_batches, _ = self.lstm_data_transform(inputs)
        count = 1

        for ipbatch in ip_batches:
            self.cleanLSTM()
            plog("Round "+str(count)," ipbatch is : ", ipbatch)

            for ip in ipbatch:
                plog("Round "+str(count)," ip is ",ip)
                output = self.goForward(np.array([ip]), train=0)

        if ipscaler and opscaler:
            print(f'Current Price : {round(ipscaler.inverse_transform(np.array([ip]))[0][0],3)} Next Price : {round(opscaler.inverse_transform(output)[0][0], 3)} \n')
            return round(opscaler.inverse_transform(output)[0][0], 3)
        else:
            print(f'input {ip} output {output}')
            return output
        
    def goValidate(self, inputs, targets, opscaler=None, ipscaler=None, filename="pred.txt"):
        plog = self.plog
        ip_batches, _ = self.lstm_data_transform(inputs)
        file = open(filename,"w")
        file.close()


        for ipbatch in ip_batches:
            self.cleanLSTM()
            #plog("Round "+str(count)," ipbatch is : ", ipbatch)
            count = 0
            for ip in ipbatch:
                target = np.array([[targets.iloc[count]['target']]])
                #plog("Round "+str(count)," ip is ",ip)
                output = self.goForward(np.array([ip]), train=0)
                if ipscaler and opscaler:
                    res = f'Current : {round(ipscaler.inverse_transform(np.array([ip]))[0][0],3)} \tTarget : {round(opscaler.inverse_transform(target)[0][0], 3)} \tPredicted : {round(opscaler.inverse_transform(output)[0][0], 3)}\n'
                    with open(filename, "a") as myfile:
                        myfile.write(res)
                    plog(res)
                else:
                    print(f'input {ip} output {output}')

                count+=1

    def saveModel(self, filename):
        with open(filename,"wb") as fp:
            p=self.getLSTMparms()
            pickle.dump(p, fp)

    def loadModel(self, filename):
        with open(filename,"rb") as fp:
            pickeledModel = pickle.load(fp)

            self.wa=pickeledModel[0]
            self.ua=pickeledModel[1]
            self.ba=pickeledModel[2]
            self.wi=pickeledModel[3]
            self.ui=pickeledModel[4]
            self.bi=pickeledModel[5]
            self.wf=pickeledModel[6]
            self.uf=pickeledModel[7]
            self.bf=pickeledModel[8]
            self.wo=pickeledModel[9]
            self.uo=pickeledModel[10]
            self.bo=pickeledModel[11]

    
## EXAMPLE CODE TO PREPARE DATASET AND RUN

# dataset=StringIO("""Date,Open,High,Low,Close,Volume,Trade_count,vwap
# 2015-12-01 09:00:00+00:00,118.88,118.94,118.88,118.94,1145,5,118.902052
# 2015-12-01 09:15:00+00:00,118.77,118.77,118.77,118.77,200,1,118.77
# 2015-12-01 09:30:00+00:00,118.69,118.69,118.6,118.6,900,4,118.61
# 2015-12-01 09:45:00+00:00,118.64,118.65,118.64,118.65,3580,5,118.648883
# 2015-12-01 10:00:00+00:00,118.65,118.65,118.55,118.55,1820,4,118.611538
# 2015-12-01 10:15:00+00:00,118.55,118.6,118.55,118.6,880,5,118.5625
# 2015-12-01 10:30:00+00:00,118.55,118.55,118.5,118.5,1878,5,118.513312
# 2015-12-01 10:45:00+00:00,118.59,118.72,118.59,118.72,2499,10,118.628431
# 2015-12-01 11:00:00+00:00,118.71,118.9,118.71,118.9,2842,11,118.86064
# 2015-12-01 11:15:00+00:00,118.87,118.87,118.87,118.87,300,2,118.87
# 2015-12-01 11:30:00+00:00,118.78,118.8,118.76,118.8,3914,22,118.785876
# 2015-12-01 11:45:00+00:00,118.8,118.99,118.77,118.9,7900,37,118.893542
# 2015-12-01 12:00:00+00:00,118.88,118.98,118.84,118.84,6540,34,118.922648
# 2015-12-01 12:15:00+00:00,118.82,118.84,118.77,118.77,5603,28,118.804962
# 2015-12-01 12:30:00+00:00,118.77,118.89,118.76,118.88,7612,31,118.824002
# """)
# ip = pd.read_table(dataset, sep=",")

# # ip=pd.read_csv('../dataset/apple_5min_data.csv')


# opscaler = MinMaxScaler()
# ipscaler = MinMaxScaler()
# inputs=ip.copy()
# inputs.drop("Date", axis=1, inplace=True)

# targets = inputs.filter(["Open"], axis=1)
# targets.columns = ['target']
# targets["target"]=targets['target'][1:].reset_index(drop=True)
# targets.iloc[-1]['target'] = targets.iloc[:-1]['target'].mean()

# inputs[['Open','High','Low','Close','Volume','Trade_count','vwap']] = ipscaler.fit_transform(inputs[['Open','High','Low','Close','Volume','Trade_count','vwap']])
# targets[['target']] = opscaler.fit_transform(targets[['target']])

# intrain, intest, optrain, optest = train_test_split(inputs, targets, test_size=0.2, shuffle=False)


# lstm = LSTM(train_data=intrain, targets=optrain, batch_size=4, debug=0, test=0)
# lstm.train(epoch=2, lr=1)
# lstm.goValidate(intest, optest, opscaler, ipscaler)
