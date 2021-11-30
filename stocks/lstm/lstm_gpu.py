# Author : Bijoy Prakash
# LSTM Model to Run on GPU using Numba

import numpy as np
from numba import jit
import pickle


def sigmoid(input):
    # Element wise sigmoid for input
    i_safe = input + 1e-12
    return 1/(1 + np.exp(-i_safe))


@jit
def init(train_data, targets, batch_size=2, debug=1, test=1):
    #3 gates and input activation
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
    # Number of features used in the input. Get it from the number of columns in the shape of the data
    numFeats = train_data.shape[1]
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

# Clean up the LSTM parameters    
@jit
def cleanLSTM(p):
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
    
# This function will go and update the LSTM parameters after the completion of the 
# backward propagation
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

# Debug function to print the LSTM parameters
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

# Transform the dataset to batches. This is needed before we start LSTM training
@jit
def lstm_data_transform(batch_size, targets, train_data=None, ip=None):
    # Format data for LSTM training in sliding window batches
    # Prepare the list for the transformed data
    X = [np.complex64(x) for x in range(0)]
    y = [np.complex64(x) for x in range(0)]
    if ip is not None:
        data = ip
    else:
        data = train_data

    # Go over the full dataset
    for i in range(data.shape[0]):
        # next sliding windows index
        end_index = i + batch_size

        # stop when index crosses the size of the dataset
        if end_index >= data.shape[0]:
            break
        # input batch
        sequence_X = data[i:end_index]
        # Input batches target is last inputs output
        sequence_y = targets[i:end_index]
        # Add the window to the list
        X.append(sequence_X)
        y.append(sequence_y)
    # Make final arrays
    ip_arr = np.array(X)
    op_arr = np.array(y)
    return ip_arr, op_arr

@jit
def goForward(p,ipt, train=1):
    #3 gates and input activation
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

    # Go here only for training. Skip for prediction
    if train:
        prev_input_activations.append(ia)
        prev_input_gates.append(input_gate)
        prev_forget_gates.append(forget_gate)
        prev_output_gates.append(output_gate)
        prev_internal_states.append(internal_state)
        prev_outputs.append(output)

    return wa, ua, ba, wi, ui, bi, wf, uf, bf, wo, uo, bo, output, internal_state, prev_input_activations, prev_input_gates, prev_forget_gates, prev_output_gates, prev_internal_states, prev_outputs, delta_op_future, der_internal_state_future, stacked_ip_weights, stacked_op_weights, input_weight_derivatives, output_weight_derivatives, bias_derivatives

#Helper function for backward propagation
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

# Backwards propagation
@jit
def travelBack(p, targets, inputs):

    # Unpack parameters
    wa, ua, ba, wi, ui, bi, wf, uf, bf, wo, uo, bo, output, internal_state, prev_input_activations, prev_input_gates, prev_forget_gates, prev_output_gates, prev_internal_states, prev_outputs, delta_op_future, der_internal_state_future, stacked_ip_weights, stacked_op_weights, input_weight_derivatives, output_weight_derivatives, bias_derivatives = p
    tempo = np.zeros((1, 1))
    loss=0

    # Go over all the targets
    for t in reversed(range(len(prev_outputs))):

        output = prev_outputs[t]
        target = targets[t]

        next_forget_gate = np.zeros((1, 1)) if (t==len(prev_outputs)-1) else prev_forget_gates[t+1]

        # Track loss
        loss = (np.power((target - output),2))/2

        # derivative of loss with respect to output
        der_loss_wrt_output = output - target

        # derivative of output
        der_output = der_loss_wrt_output + delta_op_future

        # derivative of internal state
        pog = prev_output_gates[t]
        ps = prev_internal_states[t]
        dfis = der_output * pog * (1 - (np.tanh(ps))**2 ) + (der_internal_state_future * next_forget_gate)
        der_internal_state_future = dfis

        # derivative of input activation
        pig = prev_input_gates[t]
        pia = prev_input_activations[t]
        der_input_activation = dfis * pig * (1 - pia**2)
        stacked_ders = np.copy(der_input_activation)

        # derivative of input gate
        der_inputg = dfis * pia * pig * (1 - pig)
        stacked_ders = np.row_stack((stacked_ders, der_inputg))

        # derivative of forget gate
        pps = tempo if t==0 else prev_internal_states[t-1] 
        pfg = prev_forget_gates[t]
        der_forgetg = dfis * pps * pfg * (1 - pfg)
        stacked_ders = np.row_stack((stacked_ders, der_forgetg))
        der_outputg = der_output * np.tanh(ps) * pog * (1 - pog)
        stacked_ders = np.row_stack((stacked_ders, der_outputg))

        wa, ua, ba, wi, ui, bi, wf, uf, bf, wo, uo, bo, output, internal_state, prev_input_activations, prev_input_gates, prev_forget_gates, prev_output_gates, prev_internal_states, prev_outputs, delta_op_future, der_internal_state_future, stacked_ip_weights, stacked_op_weights, input_weight_derivatives, output_weight_derivatives, bias_derivatives = stackWeights(p)

        der_input_state = np.dot(stacked_ip_weights, stacked_ders)

        # output gate derivative
        der_output_state = np.dot(stacked_op_weights, stacked_ders)
        delta_op_future = der_output_state

        der_input_weight = np.dot(stacked_ders, np.array([inputs[0][t]]))
        input_weight_derivatives += der_input_weight

        po = tempo if t==0 else prev_outputs[t-1] 
        der_op_weight = np.dot(stacked_ders, po)
        output_weight_derivatives += der_op_weight

        bias_derivatives += stacked_ders
    return wa, ua, ba, wi, ui, bi, wf, uf, bf, wo, uo, bo, output, internal_state, prev_input_activations, prev_input_gates, prev_forget_gates, prev_output_gates, prev_internal_states, prev_outputs, delta_op_future, der_internal_state_future, stacked_ip_weights, stacked_op_weights, input_weight_derivatives, output_weight_derivatives, bias_derivatives

# Model Training function
@jit
def train(p,batch_size, targets, train_data, epoch=2, lr=.01):
    #  Get prepared window sequenced data
    ip_batches, op_batches = lstm_data_transform(batch_size, targets, train_data=train_data)

    # Run it epoch number of times
    runit=1
    for runit in range (epoch):
        print("Running EPOCH ", runit)
        count = 1
        for ipbatch,opbatch in zip(ip_batches, op_batches):
            # Clean LSTM parameters after every back propagation is complete
            p=cleanLSTM(p)
            if count % 100 ==0:
                print("Running Batch", count)
            # Do forward propagation for every batch
            for ip in ipbatch:
                p1 = goForward(p,np.array([ip]))
            # Travel back and redistribute the weights
            p2 = travelBack(p1, opbatch, np.array([ipbatch]))
            # Update the LSTM parameters with the new weights adjusted by learning rate
            p3=update_lstmData(p2, lr)
            p=p3

            count+=1
    return p

# Prediction function
@jit
def goPredict(p, targets, inputs, batch_size, opscaler=None, ipscaler=None):

    wa, ua, ba, wi, ui, bi, wf, uf, bf, wo, uo, bo, output, internal_state, prev_input_activations, prev_input_gates, prev_forget_gates, prev_output_gates, prev_internal_states, prev_outputs, delta_op_future, der_internal_state_future, stacked_ip_weights, stacked_op_weights, input_weight_derivatives, output_weight_derivatives, bias_derivatives = p
    # Transform data for prediction
    ip_batches, _ = lstm_data_transform(batch_size, targets, ip=inputs)

    # Run for each window in the batch
    for ipbatch in ip_batches:
        # start with a clean LSTM model
        p=cleanLSTM(p)

        # for each batch go forward and get the output. Last batch output is the prediction
        for ip in ipbatch:
            p = goForward(p, np.array([ip]), train=0)
            wa, ua, ba, wi, ui, bi, wf, uf, bf, wo, uo, bo, output, internal_state, prev_input_activations, prev_input_gates, prev_forget_gates, prev_output_gates, prev_internal_states, prev_outputs, delta_op_future, der_internal_state_future, stacked_ip_weights, stacked_op_weights, input_weight_derivatives, output_weight_derivatives, bias_derivatives = p

    # De-Scale the output if it is requested
    if ipscaler and opscaler:
        print(f'Current Price : {round(ipscaler.inverse_transform(np.array([ip]))[0][0],3)} Next Price : {round(opscaler.inverse_transform(output)[0][0], 3)} \n')
        return round(opscaler.inverse_transform(output)[0][0], 3)
    else:
        print(f'input {ip} output {output}')
        return output

# Validation function used when splitting the model into train test slipts
@jit
def goValidate(p, inputs, targets, batch_size, opscaler=None, ipscaler=None, filename="pred.txt"):

    ip_batches, _ = lstm_data_transform(batch_size, targets, ip=inputs)
    count = 0
    for ipbatch in ip_batches:
        p=cleanLSTM(p)

        for ip in ipbatch:
            target = np.array([[targets.iloc[count]['target']]])
            p = goForward(p,np.array([ip]), train=0)
            wa, ua, ba, wi, ui, bi, wf, uf, bf, wo, uo, bo, output, internal_state, prev_input_activations, prev_input_gates, prev_forget_gates, prev_output_gates, prev_internal_states, prev_outputs, delta_op_future, der_internal_state_future, stacked_ip_weights, stacked_op_weights, input_weight_derivatives, output_weight_derivatives, bias_derivatives = p
            if ipscaler and opscaler:
                res = f'Current : {round(ipscaler.inverse_transform(np.array([ip]))[0][0],3)} \tTarget : {round(opscaler.inverse_transform(target)[0][0], 3)} \tPredicted : {round(opscaler.inverse_transform(output)[0][0], 3)}\n'
                print(res)
            else:
                print(f'input {ip} output {output}')

        count+=1

# Pickle and save a copy of the model. Saves a copy of the LSTM parameters
def saveModel(filename, p):
    with open(filename,"wb") as fp:
        pickle.dump(p, fp)

# Loads a pickled copy of the LSTM parameters from a local file and initializes the model
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