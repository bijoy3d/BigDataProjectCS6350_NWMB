import numpy as np
import copy
import os
from numba import jit, cuda

from io import StringIO
import pandas as pd
import pickle
import matplotlib.pyplot as plt


def sigmoid(x):
    """
    Computes the element-wise sigmoid activation function for an array x.

    Args:
     `x`: the array where the function is applied
     `derivative`: if set to True will return the derivative instead of the forward pass
    """
    x_safe = x + 1e-12
    return 1 / (1 + np.exp(-x_safe))


@jit
def init(train_data, targets, batch_size=2, debug=1, test=1):
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
    test = test

    # The number of records that would go inside the LSTM at one time. A sequence of records.
    batch_size = batch_size
    numFeats = train_data.shape[1] ###### CHANGE IT TO GET DYNAMICALLY FROM INPUT
    # Enable debug logs
    debug = debug

    # Training Data
    train_data = train_data
    # Target of training data
    targets = targets


    # input_activation
    wa = np.random.random((numFeats, 1))
    ua = np.random.random((1, 1))
    ba = np.random.random((1, 1))

    # input_gate
    wi = np.random.random((numFeats, 1))
    ui = np.random.random((1, 1))
    bi = np.random.random((1, 1))

    # forget_gate
    wf = np.random.random((numFeats, 1))
    uf = np.random.random((1, 1))
    bf = np.random.random((1, 1))

    # output_gate
    wo = np.random.random((numFeats, 1))
    uo = np.random.random((1, 1))
    bo = np.random.random((1, 1))
    # Forward Propogation Parameters
    prev_input_activation = 0
    prev_input_gate = 0
    prev_forget_gate = 0
    prev_output_gate = 0

    input_activation = 0
    input_gate = 0
    forget_gate = 0
    output_gate = 0
    internal_state = np.zeros((1, 1))
    output = np.zeros((1, 1))

    prev_input_activations = [np.complex64(x) for x in range(0)]
    prev_input_gates  = [np.complex64(x) for x in range(0)]
    prev_output_gates  = [np.complex64(x) for x in range(0)]
    prev_forget_gates  = [np.complex64(x) for x in range(0)]
    prev_internal_states  = [np.complex64(x) for x in range(0)]
    prev_outputs = [np.complex64(x) for x in range(0)]


    # Backward Propogation Parameters
    stacked_ip_weights = [np.complex64(x) for x in range(0)]
    stacked_op_weights = [np.complex64(x) for x in range(0)]

    der_internal_state_future = np.zeros((1, 1))
    delta_op_future = np.zeros((1, 1))

    input_weight_derivatives = 0
    output_weight_derivatives = 0
    bias_derivatives = 0
    # Clean and init LSTM
    #cleanLSTM()


    if test:
        batchSize = 1
        wa[0]=0.45
        wa[1]=0.25
        ua[0]=0.15
        ba[0]=0.2
        wi[0]=0.95
        wi[1]=0.8
        ui[0]=0.8
        bi[0]=0.65
        wf[0]=0.7
        wf[1]=0.45
        uf[0]=0.1
        bf[0]=0.15
        wo[0]=0.6
        wo[1]=0.4
        uo[0]=0.25
        bo[0]=0.1
    return wa, ua, ba, wi, ui, bi, wf, uf, bf, wo, uo, bo, output, internal_state, prev_input_activations, prev_input_gates, prev_forget_gates, prev_output_gates, prev_internal_states, prev_outputs, delta_op_future, der_internal_state_future, stacked_ip_weights, stacked_op_weights, input_weight_derivatives, output_weight_derivatives, bias_derivatives

@jit
def cleanLSTM(p):
    # try:
    #     wa, ua, ba, wi, ui, bi, wf, uf, bf, wo, uo, bo, output, internal_state, prev_input_activations, prev_input_gates, prev_forget_gates, prev_output_gates, prev_internal_states, prev_outputs, delta_op_future, der_internal_state_future, stacked_ip_weights, stacked_op_weights, input_weight_derivatives, output_weight_derivatives, bias_derivatives = p
    # except:
    wa = p[0]
    ua = p[1]
    ba = p[2]
    wi = p[3]
    ui = p[4]
    bi = p[5]
    wf = p[6]
    uf = p[7]
    bf = p[8]
    wo = p[9]
    uo = p[10]
    bo = p[11]
    # Forward Propogation Parameters
    prev_input_activation = 0
    prev_input_gate = 0
    prev_forget_gate = 0
    prev_output_gate = 0

    input_activation = 0
    input_gate = 0
    forget_gate = 0
    output_gate = 0
    internal_state = np.zeros((1, 1))
    output = np.zeros((1, 1))

    prev_input_activations = [np.complex64(x) for x in range(0)]
    prev_input_gates  = [np.complex64(x) for x in range(0)]
    prev_output_gates  = [np.complex64(x) for x in range(0)]
    prev_forget_gates  = [np.complex64(x) for x in range(0)]
    prev_internal_states  = [np.complex64(x) for x in range(0)]
    prev_outputs = [np.complex64(x) for x in range(0)]


    # Backward Propogation Parameters
    stacked_ip_weights = [np.complex64(x) for x in range(0)]
    stacked_op_weights = [np.complex64(x) for x in range(0)]

    der_internal_state_future = np.zeros((1, 1))
    delta_op_future = np.zeros((1, 1))

    input_weight_derivatives = 0
    output_weight_derivatives = 0
    bias_derivatives = 0
    return wa, ua, ba, wi, ui, bi, wf, uf, bf, wo, uo, bo, output, internal_state, prev_input_activations, prev_input_gates, prev_forget_gates, prev_output_gates, prev_internal_states, prev_outputs, delta_op_future, der_internal_state_future, stacked_ip_weights, stacked_op_weights, input_weight_derivatives, output_weight_derivatives, bias_derivatives
    
@jit
def update_lstmData(p, lr=.01):
    wa, ua, ba, wi, ui, bi, wf, uf, bf, wo, uo, bo, output, internal_state, prev_input_activations, prev_input_gates, prev_forget_gates, prev_output_gates, prev_internal_states, prev_outputs, delta_op_future, der_internal_state_future, stacked_ip_weights, stacked_op_weights, input_weight_derivatives, output_weight_derivatives, bias_derivatives = p
    dip = input_weight_derivatives
    dop = output_weight_derivatives
    db = bias_derivatives

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
    wa=wa
    ua=ua
    ba=ba
    wi=wi
    ui=ui
    bi=bi
    wf=wf
    uf=uf
    bf=bf
    wo=wo
    uo=uo
    bo=bo
    return wa, ua, ba, wi, ui, bi, wf, uf, bf, wo, uo, bo, output, internal_state, prev_input_activations, prev_input_gates, prev_forget_gates, prev_output_gates, prev_internal_states, prev_outputs, delta_op_future, der_internal_state_future, stacked_ip_weights, stacked_op_weights, input_weight_derivatives, output_weight_derivatives, bias_derivatives

def printLSTMparms():
    lstmData = getLSTMparms()
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

@jit
def lstm_data_transform(batch_size, targets, train_data=None, ip=None):
    """ Changes data to the format for LSTM training 
for sliding window approach """
    # Prepare the list for the transformed data
    X = [np.complex64(x) for x in range(0)]
    y = [np.complex64(x) for x in range(0)]
    if ip is not None:
        data = ip
    else:
        data = train_data
    # Loop of the entire data set
    for i in range(data.shape[0]):
        # compute a new (sliding window) index
        end_ix = i + batch_size

        # if index is larger than the size of the dataset, we stop
        if end_ix >= data.shape[0]:
            break
        # Get a sequence of data for x
        seq_X = data[i:end_ix]
        # Get only the last element of the sequency for y
        seq_y = targets[i:end_ix]
        # Append the list with sequencies
        X.append(seq_X)
        y.append(seq_y)
    # Make final arrays
    x_array = np.array(X)
    y_array = np.array(y)
    return x_array, y_array

def plog(*msg, f=0):
    debug=0
    if debug or f:
        print(*msg)

def setLSTMparms(parms):
    wa, ua, ba, wi, ui, bi, wf, uf, bf, wo, uo, bo = parms

def getLSTMparms():
    return wa, ua, ba, wi, ui, bi, wf, uf, bf, wo, uo, bo

@jit
def goForward(p,ipt, train=1):
    #4 gates
    # input_activation = tanh(wa [inner] input + ua [inner] prev_output + ba)
    # input_gate  = sigmoid(wi [inner] input + ui [inner] prev_output + bi)
    # forget_gate = sigmoid(wf [inner] input + uf [inner] prev_output + bf)
    # output_gate = sigmoid(wo [inner] input + uo [inner] prev_output + bo)

    # 2 states
    # internal_state = (input_activation [Element wise] input_gate) + (forget_gate [Element wise] prev_internal_state)
    # output = tanh(internal_state) [Element wise] output_gate

    wa, ua, ba, wi, ui, bi, wf, uf, bf, wo, uo, bo, output, internal_state, prev_input_activations, prev_input_gates, prev_forget_gates, prev_output_gates, prev_internal_states, prev_outputs, delta_op_future, der_internal_state_future, stacked_ip_weights, stacked_op_weights, input_weight_derivatives, output_weight_derivatives, bias_derivatives = p

    po = output
    ps = internal_state
    #plog("wa : ", wa.T)
    #plog("ipt : ", ipt)
    #plog("ua : ", ua)
    #plog("po : ", po)
    #plog("ba : ", ba)

    #plog("ipt T shape", ipt.T.shape)
    #plog("po shape", po.shape)

       # print("incoming input = ",ippo.T)

    input_plus_prev_output = np.row_stack((ipt.T, po))
    ippo = input_plus_prev_output


    # input activation
    input_activation = np.tanh((np.inner(wa.T, ipt)) + (np.inner(ua, po)) + ba)
    ia = input_activation

    # input gate
    input_gate = sigmoid((np.inner(wi.T, ipt)) + (np.inner(ui, po)) + bi)

    # forget gate
    forget_gate = sigmoid((np.inner(wf.T, ipt)) + (np.inner(uf, po)) + bf)

    # output gate
    output_gate = sigmoid((np.inner(wo.T, ipt)) + (np.inner(uo, po)) + bo)

    # internal state
    internal_state = (np.multiply(ia, input_gate)) + (np.multiply(forget_gate, ps))

    # output
    output = np.multiply(np.tanh(internal_state), output_gate)

    if train:
        prev_input_activations.append(ia)
        prev_input_gates.append(input_gate)
        prev_forget_gates.append(forget_gate)
        prev_output_gates.append(output_gate)
        prev_internal_states.append(internal_state)
        prev_outputs.append(output)

    #plog("input_activation = ",ia)
    #plog("input gate : ", input_gate)
    #plog("forget gate : ", forget_gate)
    #plog("output gate : ",output_gate)
    #plog("internal state", internal_state)
    #plog("output = ",output)
    #plog("----------------------------------")
    return wa, ua, ba, wi, ui, bi, wf, uf, bf, wo, uo, bo, output, internal_state, prev_input_activations, prev_input_gates, prev_forget_gates, prev_output_gates, prev_internal_states, prev_outputs, delta_op_future, der_internal_state_future, stacked_ip_weights, stacked_op_weights, input_weight_derivatives, output_weight_derivatives, bias_derivatives

@jit
def stackWeights(p):
    wa, ua, ba, wi, ui, bi, wf, uf, bf, wo, uo, bo, output, internal_state, prev_input_activations, prev_input_gates, prev_forget_gates, prev_output_gates, prev_internal_states, prev_outputs, delta_op_future, der_internal_state_future, stacked_ip_weights, stacked_op_weights, input_weight_derivatives, output_weight_derivatives, bias_derivatives = p
    stacked_ip_weights = np.copy(wa)
    stacked_ip_weights = np.column_stack((stacked_ip_weights, wi))
    stacked_ip_weights = np.column_stack((stacked_ip_weights, wf))
    stacked_ip_weights = np.column_stack((stacked_ip_weights, wo))
    stacked_ip_weights = stacked_ip_weights

    stacked_op_weights = np.copy(ua)
    stacked_op_weights = np.column_stack((stacked_op_weights, ui))
    stacked_op_weights = np.column_stack((stacked_op_weights, uf))
    stacked_op_weights = np.column_stack((stacked_op_weights, uo))
    stacked_op_weights = stacked_op_weights

    return wa, ua, ba, wi, ui, bi, wf, uf, bf, wo, uo, bo, output, internal_state, prev_input_activations, prev_input_gates, prev_forget_gates, prev_output_gates, prev_internal_states, prev_outputs, delta_op_future, der_internal_state_future, stacked_ip_weights, stacked_op_weights, input_weight_derivatives, output_weight_derivatives, bias_derivatives

@jit
def travelBack(p, targets, inputs):

    # Unpack parameters
    wa, ua, ba, wi, ui, bi, wf, uf, bf, wo, uo, bo, output, internal_state, prev_input_activations, prev_input_gates, prev_forget_gates, prev_output_gates, prev_internal_states, prev_outputs, delta_op_future, der_internal_state_future, stacked_ip_weights, stacked_op_weights, input_weight_derivatives, output_weight_derivatives, bias_derivatives = p
    tempo = np.zeros((1, 1))
    loss=0
    #plog("Targets is",targets)
    #plog("Inputs is",inputs)

    for t in reversed(range(len(prev_outputs))):

        output = prev_outputs[t]
        target = targets[t]

        next_forget_gate = np.zeros((1, 1)) if (t==len(prev_outputs)-1) else prev_forget_gates[t+1]

        #plog("previous outputs = ", str(prev_outputs))
        #plog("target = ",str(target))
        #plog("output = ", str(output))

        # Track loss
        loss = (np.power((target - output),2))/2
        #plog("loss = ", str(loss), f=0)

        # derivative of loss with respect to output
        der_loss_wrt_output = output - target
        #plog("der_loss_wrt_output = ", der_loss_wrt_output)

        # derivative of output
        der_output = der_loss_wrt_output + delta_op_future
        #plog("der_output = ", der_output)

        # derivative of internal state
        pog = prev_output_gates[t]
        ps = prev_internal_states[t]
        dfis = der_output * pog * (1 - (np.tanh(ps))**2 ) + (der_internal_state_future * next_forget_gate)
        der_internal_state_future = dfis
        #plog("der internal state = ", dfis)
        #plog("pog : ", pog)
        #plog("ps : ", ps)


        pig = prev_input_gates[t]
        pia = prev_input_activations[t]
        der_input_activation = dfis * pig * (1 - pia**2)
        #plog("der_input_activation = ", der_input_activation)
        stacked_ders = np.copy(der_input_activation)

        der_inputg = dfis * pia * pig * (1 - pig)
        stacked_ders = np.row_stack((stacked_ders, der_inputg))
        #plog("der_input = ", der_inputg)

        pps = tempo if t==0 else prev_internal_states[t-1] 
        pfg = prev_forget_gates[t]
        der_forgetg = dfis * pps * pfg * (1 - pfg)
        stacked_ders = np.row_stack((stacked_ders, der_forgetg))
        #plog("der_forget = ", der_forgetg)   

        #plog("pps : ", pps, t-1)
        #plog("pfg : ", pfg)
        #plog("dfis : ", str(dfis))

        der_outputg = der_output * np.tanh(ps) * pog * (1 - pog)
        stacked_ders = np.row_stack((stacked_ders, der_outputg))
        #plog("der_output = ", der_outputg)

        wa, ua, ba, wi, ui, bi, wf, uf, bf, wo, uo, bo, output, internal_state, prev_input_activations, prev_input_gates, prev_forget_gates, prev_output_gates, prev_internal_states, prev_outputs, delta_op_future, der_internal_state_future, stacked_ip_weights, stacked_op_weights, input_weight_derivatives, output_weight_derivatives, bias_derivatives = stackWeights(p)

        der_input_state = np.dot(stacked_ip_weights, stacked_ders)
        #plog("der_input_state = ", der_input_state)

        der_output_state = np.dot(stacked_op_weights, stacked_ders)
        #plog("der_output_state = ", der_output_state)
        delta_op_future = der_output_state

        #plog("inputs t is : ",str(t), np.array([inputs[0][t]]))
        der_input_weight = np.dot(stacked_ders, np.array([inputs[0][t]]))
        input_weight_derivatives += der_input_weight
        #plog("der_input_weight : ", der_input_weight)

        po = tempo if t==0 else prev_outputs[t-1] 
        der_op_weight = np.dot(stacked_ders, po)
        output_weight_derivatives += der_op_weight
        #plog("der_op_weight : ", der_op_weight)

        bias_derivatives += stacked_ders
    return wa, ua, ba, wi, ui, bi, wf, uf, bf, wo, uo, bo, output, internal_state, prev_input_activations, prev_input_gates, prev_forget_gates, prev_output_gates, prev_internal_states, prev_outputs, delta_op_future, der_internal_state_future, stacked_ip_weights, stacked_op_weights, input_weight_derivatives, output_weight_derivatives, bias_derivatives

@jit
def train(p,batch_size, targets, train_data, epoch=2, lr=.01):
    ip_batches, op_batches = lstm_data_transform(batch_size, targets, train_data=train_data)

    #print(op_batches)
    runit=1
    for runit in range (epoch):
        print("Running EPOCH ", runit)
        count = 1
        for ipbatch,opbatch in zip(ip_batches, op_batches):
            p=cleanLSTM(p)
            if count % 100 ==0:
                print("Round ipbatch", count)
            for ip in ipbatch:
                p1 = goForward(p,np.array([ip]))
            p2 = travelBack(p1, opbatch, np.array([ipbatch]))
            p3=update_lstmData(p2, lr)
            p=p3

            count+=1
    return p

@jit
def goPredict(p, targets, inputs, batch_size, opscaler=None, ipscaler=None):

    wa, ua, ba, wi, ui, bi, wf, uf, bf, wo, uo, bo, output, internal_state, prev_input_activations, prev_input_gates, prev_forget_gates, prev_output_gates, prev_internal_states, prev_outputs, delta_op_future, der_internal_state_future, stacked_ip_weights, stacked_op_weights, input_weight_derivatives, output_weight_derivatives, bias_derivatives = p
    ip_batches, _ = lstm_data_transform(batch_size, targets, ip=inputs)
    count = 1

    for ipbatch in ip_batches:
        p=cleanLSTM(p)
        #plog("Round "+str(count)," ipbatch is : ", ipbatch)

        for ip in ipbatch:
            print("Round "+str(count))
            p = goForward(p, np.array([ip]), train=0)
            wa, ua, ba, wi, ui, bi, wf, uf, bf, wo, uo, bo, output, internal_state, prev_input_activations, prev_input_gates, prev_forget_gates, prev_output_gates, prev_internal_states, prev_outputs, delta_op_future, der_internal_state_future, stacked_ip_weights, stacked_op_weights, input_weight_derivatives, output_weight_derivatives, bias_derivatives = p
            if ipscaler and opscaler:
                print(f'Current Price : {round(ipscaler.inverse_transform(np.array([ip]))[0][0],3)} \
                        Next Price : {round(opscaler.inverse_transform(output)[0][0], 3)} \n')
            else:
                print(f'input {ip} output {output}')

        count+=1
    if ipscaler and opscaler:
        print(f'Current Price : {round(ipscaler.inverse_transform(np.array([ip]))[0][0],3)} Next Price : {round(opscaler.inverse_transform(output)[0][0], 3)} \n')
        return round(opscaler.inverse_transform(output)[0][0], 3)
    else:
        print(f'input {ip} output {output}')
        return output

@jit
def goValidate(p, inputs, targets, batch_size, opscaler=None, ipscaler=None, filename="pred.txt"):

    ip_batches, _ = lstm_data_transform(batch_size, targets, ip=inputs)
#     file = open(filename,"w")
#     file.close()
#    print(ip_batches)
    count = 0
    for ipbatch in ip_batches:
        p=cleanLSTM(p)

        for ip in ipbatch:
#            print(ip)
            target = np.array([[targets.iloc[count]['target']]])
            ##plog("Round "+str(count)," ip is ",ip)
            p = goForward(p,np.array([ip]), train=0)
            wa, ua, ba, wi, ui, bi, wf, uf, bf, wo, uo, bo, output, internal_state, prev_input_activations, prev_input_gates, prev_forget_gates, prev_output_gates, prev_internal_states, prev_outputs, delta_op_future, der_internal_state_future, stacked_ip_weights, stacked_op_weights, input_weight_derivatives, output_weight_derivatives, bias_derivatives = p
            if ipscaler and opscaler:
                res = f'Current : {round(ipscaler.inverse_transform(np.array([ip]))[0][0],3)} \tTarget : {round(opscaler.inverse_transform(target)[0][0], 3)} \tPredicted : {round(opscaler.inverse_transform(output)[0][0], 3)}\n'
                print(res)
#                 with open(filename, "a") as myfile:
#                     myfile.write(res)
                #plog(res)
            else:
                print(f'input {ip} output {output}')

        count+=1

def saveModel(filename, p):
    with open(filename,"wb") as fp:
        pickle.dump(p, fp)

def loadModel(filename):
    with open(filename,"rb") as fp:
        pickeledModel = pickle.load(fp)

        wa=pickeledModel[0]
        ua=pickeledModel[1]
        ba=pickeledModel[2]
        wi=pickeledModel[3]
        ui=pickeledModel[4]
        bi=pickeledModel[5]
        wf=pickeledModel[6]
        uf=pickeledModel[7]
        bf=pickeledModel[8]
        wo=pickeledModel[9]
        uo=pickeledModel[10]
        bo=pickeledModel[11]

        return wa, ua, ba, wi, ui, bi, wf, uf, bf, wo, uo, bo