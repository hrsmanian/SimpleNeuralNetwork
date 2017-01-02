import numpy as np;
import math;
import random;

    
""" 
A simple neural network class that can handle multiple hidden layers
and trained using stochastic gradient descent

Supports Relu, Tanh, sigmoid activations for hidden layers
Supports Sigmoid and softmax activations for output layers

"""
class NN :
    def sigmoid(self, x):
        return 1. / (1 + np.exp(-x))

    def dsigmoid(self, x):
        return x * (1. - x)

    def tanh(self, x):
        return np.tanh(x)

    def dtanh(self, x):
        return 1. - x * x

    def softmax(self, x):
        shiftx = x - np.max(x)
        exps = np.exp(shiftx)
        return exps / np.sum(exps)

    def relu(self, x) :
        return x * (x > 0)

    def dReLU(self, x):
        return 1. * (x > 0)

    def oGradCrossEntropy(self, outputs, targets) :
        return targets - outputs

    def oGradMSE(self, outputs, targets) :
        oDerivative = (1 - outputs) * outputs;
        return oDerivative * (targets - outputs)

    def __init__(self, numInput, layers, costFunc = 'crossEntropyError') :
        self.numInput = numInput;
        # self.numOutput = numOutput;
        self.numHidden = len(layers) - 1

      
        self.weights = []
        self.biases = []
        self.activations = []
        self.outputs = []
        self.derivatives = []

        self.rnd = random.Random(0)
        lo = -0.01
        hi = 0.01
        numCols = numInput;

        # input layer is all dummy for convenience
        # and similarity
        #
        self.weights.append(np.empty((0,0)))
        self.biases.append(np.zeros(numInput))
        self.activations.append(None)
        self.outputs.append(np.zeros(numInput))
        self.derivatives.append(None)

        # hidden and output layer
        #
        for idx, (numNodes, activationFunc) in enumerate(layers) :
            numRows = numNodes;
            ihWeights = np.zeros((numRows, numCols));
            
            for i in range(numRows):
                for j in range(numCols):
                    ihWeights[i][j] = (hi - lo) * self.rnd.random() + lo
            self.weights.append(ihWeights);

            bias = np.zeros(numRows)
            for i in range(numRows):
                bias[i] = (hi - lo) * self.rnd.random() + lo
            self.biases.append(bias)
            numCols = numRows

            outputs = np.zeros(numRows)
            self.outputs.append(outputs)

            if (idx != (len(layers) - 1)) :
                if (activationFunc == 'relu') :
                    self.activations.append(self.relu)
                    self.derivatives.append(self.dReLU)

                elif (activationFunc == 'tanh') :
                    self.activations.append(self.tanh)
                    self.derivatives.append(self.dtanh)

                else :
                    self.activations.append(self.sigmoid)
                    self.derivatives.append(self.dsigmoid)
            else :
                # output layer, currently supports only softmax or sigmoid activation
                # cost function needs to be 'mse' or 'crossEntropyError'
                #
                self.numOutput = numNodes;
                if (activationFunc == 'softmax') :
                    self.activations.append(self.softmax)
                else :
                    self.activations.append(self.sigmoid)

                if (costFunc == 'mse') :
                    self.costFunction = 'mse'
                    self.derivatives.append(self.oGradMSE)
                else :
                    self.costFunction = 'crossEntropyError'
                    self.derivatives.append(self.oGradCrossEntropy)


    # compute outputs given the input
    # input is copied over as output of the first (input) layer
    #
    def computeOutputs(self, input) :

        self.outputs[0] = input
        for i in range(1, len(self.outputs)) :
            output = np.dot(self.weights[i], self.outputs[i-1]) + self.biases[i]
            self.outputs[i] = self.activations[i](output)

        return self.outputs[len(self.outputs) - 1]

    # compute mse of the whole data
    #
    def computeMse(self, inputData, inputLabels) :

        mse = 0.0
        for ii in range(len(inputData)) :
            tValues = inputLabels[ii][:self.numOutput]
            output = self.computeOutputs(inputData[ii][:]);
            mse += np.sum(np.square(output - tValues)) / 2.0;

        mse = mse / len(inputData);

        return mse;

    # compute ce of the whole data
    #
    def computeCrossEntropyerror(self, inputData, inputLabels) :

        error = 0.0
        for ii in range(len(inputData)) :
            tValues = inputLabels[ii][:self.numOutput]
            maxIndex = np.argmax(tValues)
            output = self.computeOutputs(inputData[ii][:]);
            error += - (np.log(output[maxIndex]))            

        error = error / len(inputData);

        return error;

    # train using stochastic gradient descent
    #
    def train(self, train_data, train_labels, max_epochs, learnRate, momentum) :
        errorGrads = []
        prevWeightsDelta = []
        prevBiasesDelta = []

        for i in range(len(self.biases)) :
            grads = np.zeros(len(self.biases[i]))
            errorGrads.append(grads)

            prevWeights = np.zeros(self.weights[i].shape)
            prevWeightsDelta.append(prevWeights)

            prevBias = np.zeros(len(self.biases[i]))
            prevBiasesDelta.append(prevBias)

        epoch = 0;
        sequence = [i for i in range(len(train_data))]
        
        while epoch < max_epochs:
            print("Starting Epoch:" + str(epoch));
            self.rnd.shuffle(sequence)
            for ii in range(len(train_data)):
                idx = sequence[ii];
                t_values = train_labels[idx][:self.numOutput]

                self.computeOutputs(train_data[idx][:]) # outputs stored internally

                # output layer handled differently
                # the gradient is dependent upon whether the error function is mse or cross entropy
                # it is also dependent on the activation function but since we support only softmax 
                # and sigmoid for output activation, both have same derivatives
                #
                index = len(self.outputs) - 1;
                errorGrads[index] = self.derivatives[index](self.outputs[index], t_values)

                # loop over hidden layers backwards
                # we loop in reverse because the gradient is dependent on the next layer
                #
                for i in range(len(errorGrads) - 2, 0, -1) :
                    hDerivative = self.derivatives[i](self.outputs[i])
                    hSums = np.dot(self.weights[i+1].T, errorGrads[i+1])
                    errorGrads[i] = hDerivative * hSums

                # update weights and biases
                # can be done in any order
                #
                for i in range(1, len(errorGrads)) :
                    weightDeltas = np.outer(errorGrads[i], self.outputs[i-1])
                    weightDeltas = learnRate * weightDeltas
                    self.weights[i] = self.weights[i] + weightDeltas
                    self.weights[i] = self.weights[i] + (momentum * prevWeightsDelta[i]);
                    prevWeightsDelta[i] = weightDeltas;

                    biasDeltas = learnRate * errorGrads[i]
                    self.biases[i] = self.biases[i] + biasDeltas
                    self.biases[i] = self.biases[i] + (momentum * prevBiasesDelta[i]);
                    prevBiasesDelta[i] = biasDeltas;

                # print("Data:" + str(ii));

            # mse = self.computeMse(train_data, train_labels)
            mse = self.computeCrossEntropyerror(train_data, train_labels)
            print("Ending Epoch:" + str(epoch) + " mse:" + str(mse));
            epoch += 1;

    # calculate the overall accuracy given the 
    # data and labels
    #
    def accuracy(self, data, labels):
        num_correct = 0
        num_wrong = 0

        for i in range(len(data)):
            tValues = labels[i][:self.numOutput]

            yValues = self.computeOutputs(data[i][:])
            maxIndex = np.argmax(yValues)

            if tValues[maxIndex] == 1.0:
                num_correct += 1
            else:
                num_wrong += 1

        return (num_correct * 1.0) / (num_correct + num_wrong)
