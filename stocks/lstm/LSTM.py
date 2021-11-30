#!/usr/bin/env python
# coding: utf-8
# Author : Bijoy Prakash

import numpy as np
import pickle

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
        # 3 gates
        # 
        # input_gate  = sigmoid(wi [inner] input + ui [inner] prev_output + bi)
        # forget_gate = sigmoid(wf [inner] input + uf [inner] prev_output + bf)
        # output_gate = sigmoid(wo [inner] input + uo [inner] prev_output + bo)
        # Input activation
        # input_activation = tanh(wa [inner] input + ua [inner] prev_output + ba)
        # 2 states
        # internal_state = (input_activation [Element wise] input_gate) + (forget_gate [Element wise] prev_internal_state)
        # output = tanh(internal_state) [Element wise] output_gate

        # Batch size should always be less than the total size of the dataset
        if batch_size >= len(train_data):
            print(f'Batch Size {batch_size} should be less than the size of the dataset {len(train_data)}')
            return None
        
        # To enable test mode. Batch size would be set as 2
        self.test = test
        
        # The number of records that would go inside the LSTM at one time. A sequence of records.
        self.batch_size = batch_size
        # Number of features used in the input. Get it from the number of columns in the shape of the data
        numFeats = train_data.shape[1]
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


        # Static input for test mode
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

    
    # Clean up the LSTM parameters    
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

    # This function will go and update the LSTM parameters after the completion of the 
    # backward propagation
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

    # Debug function to print the LSTM parameters
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


    # Transform the dataset to batches. This is needed before we start LSTM training
    def lstm_data_transform(self, ip=None):
        # Format data for LSTM training in sliding window batches
        # Prepare the list for the transformed data
        X, y = list(), list()
        if ip is not None:
            data = ip
        else:
            data = self.train_data

        # Go over the full dataset
        for i in range(data.shape[0]):
            # next sliding windows index
            end_index = i + self.batch_size

            # stop when index crosses the size of the dataset
            if end_index >= data.shape[0]:
                break

            # input batch
            sequence_X = data[i:end_index]
            # Input batches target is last inputs output
            sequence_y = self.targets[i:end_index]
            # Add the window to the list
            X.append(sequence_X)
            y.append(sequence_y)
        
        # Final dataset for training
        ip_array = np.array(X)
        op_array = np.array(y)
        return ip_array, op_array


    # Function for debugging
    def plog(self, *msg, f=0):
        if self.debug or f:
            print(*msg)

    # Set all LSTM parameters
    def setLSTMparms(self, parms):
        self.wa, self.ua, self.ba, self.wi, self.ui, self.bi, self.wf, self.uf, self.bf, self.wo, self.uo, self.bo = parms
        
    # Get all LSTM parameters
    def getLSTMparms(self):
        return self.wa, self.ua, self.ba, self.wi, self.ui, self.bi, self.wf, self.uf, self.bf, self.wo, self.uo, self.bo

    # Forward Propogation
    def goForward(self, ipt, train=1):
        #3 gates and input activation
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
        
        # Enter this loop only if it is training. If we are predicting then just skip this and return output
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
        
    # Weight stack helper function
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

    # Backwards propagation
    def travelBack(self, targets, inputs):

        plog = self.plog
        wa, ua, ba, wi, ui, bi, wf, uf, bf, wo, uo, bo = self.getLSTMparms()
        tempo = np.zeros((1, 1))
        loss=0
        plog("Targets is",targets)
        plog("Inputs is",inputs)

        # Go over all input batches
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

            # derivative of input activation
            pig = self.prev_input_gates[t]
            pia = self.prev_input_activations[t]
            der_input_activation = dfis * pig * (1 - pia**2)
            plog("der_input_activation = ", der_input_activation)
            stacked_ders = np.copy(der_input_activation)

            # input  gate derivative
            der_inputg = dfis * pia * pig * (1 - pig)
            stacked_ders = np.row_stack((stacked_ders, der_inputg))
            plog("der_input = ", der_inputg)

            # forget gate derivative
            pps = tempo if t==0 else self.prev_internal_states[t-1] 
            pfg = self.prev_forget_gates[t]
            der_forgetg = dfis * pps * pfg * (1 - pfg)
            stacked_ders = np.row_stack((stacked_ders, der_forgetg))
            plog("der_forget = ", der_forgetg)   
            
            plog("pps : ", pps, t-1)
            plog("pfg : ", pfg)
            plog("dfis : ", str(dfis))

            # output gate derivative
            der_outputg = der_output * np.tanh(ps) * pog * (1 - pog)
            stacked_ders = np.row_stack((stacked_ders, der_outputg))
            plog("der_output = ", der_outputg)

            self.stackWeights()


            # input state derivative
            der_input_state = np.dot(self.stacked_ip_weights, stacked_ders)
            plog("der_input_state = ", der_input_state)

            # output state derivative
            der_output_state = np.dot(self.stacked_op_weights, stacked_ders)
            plog("der_output_state = ", der_output_state)
            self.delta_op_future = der_output_state

            # weight redistribution
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
    
    # Model Training function
    def train(self, epoch=2, lr=.01):
        plog = self.plog

        # Get prepared window sequenced data
        ip_batches, op_batches = self.lstm_data_transform()
        
        # Run it epoch number of times
        for runit in range (epoch):
            print("Running EPOCH ", runit+1)
            count = 1
            for ipbatch,opbatch in zip(ip_batches, op_batches):
                # Clean LSTM parameters after every back propagation is complete
                self.cleanLSTM()
                if count % 100 ==0:
                    print("Runnign Batch "+str(count))
                # Do forward propagation for every batch
                for ip in ipbatch:
                    plog("Round "+str(count)," ip is ",ip)
                    self.goForward(np.array([ip]))
                # Travel back and redistribute the weights
                loss = self.travelBack(opbatch, np.array([ipbatch]))
                # Update the LSTM parameters with the new weights adjusted by learning rate
                self.update_lstmData(lr)
                count+=1
            print("loss at epoch",runit+1,"is ", loss)
    
    # Prediction function
    def goPredict(self, inputs, opscaler=None, ipscaler=None):
        plog = self.plog
        # Transform data for prediction
        ip_batches, _ = self.lstm_data_transform(inputs)
        count = 1

        # Run for each window in the batch
        for ipbatch in ip_batches:
            # start with a clean LSTM model
            self.cleanLSTM()
            plog("Round "+str(count)," ipbatch is : ", ipbatch)

            # for each batch go forward and get the output. Last batch output is the prediction
            for ip in ipbatch:
                plog("Round "+str(count)," ip is ",ip)
                output = self.goForward(np.array([ip]), train=0)

        # De-Scale the output if it is requested
        if ipscaler and opscaler:
            print(f'Current Price : {round(ipscaler.inverse_transform(np.array([ip]))[0][0],3)} Next Price : {round(opscaler.inverse_transform(output)[0][0], 3)} \n')
            return round(opscaler.inverse_transform(output)[0][0], 3)
        else:
            print(f'input {ip} output {output}')
            return output
        
    # Validation function used when splitting the model into train test slipts
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

    # Pickle and save a copy of the model. Saves a copy of the LSTM parameters
    def saveModel(self, filename):
        with open(filename,"wb") as fp:
            p=self.getLSTMparms()
            pickle.dump(p, fp)

    # Loads a pickled copy of the LSTM parameters from a local file and initializes the model
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