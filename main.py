import numpy as np;
from NN import NN
from sys import exit

def ReadData(inputFileName) :

    count = 0;
    numData = 0;
    numInput = 0;
    numOutput = 0;
    inputListOfList = [];
    outputListOfList = [];

    with open (inputFileName, "r") as fpIn :
        while True :
            line = fpIn.readline();
            if not line :
                break;

            if (count == 0) :
                cols = line.strip().split(",");
                numData = int(cols[0]);
                numInput = int(cols[1]);
                numOutput = int(cols[2]);

                count += 1;

            else :
                cols = line.strip().split(",");
                inList = list(map(float, cols));
                inList[:] = [x / 255.0 for x in inList];
                inputListOfList.append(inList);

                line = fpIn.readline();
                if not line :
                    break;

                cols = line.strip().split(",");
                outputListOfList.append(list(map(float, cols)));

                count += 2;
    return (numData, numInput, numOutput, np.array(inputListOfList), np.array(outputListOfList));

testNumData, testNumInput, testNumOutput, testData, testLabels = ReadData("mnist_test_encoded.csv");
trainNumData, trainNumInput, trainNumOutput, trainData, trainLabels = ReadData("mnist_train_encoded.csv");

if (trainNumInput != testNumInput or
    trainNumOutput != testNumOutput) :
    print("Dimension mismatch. Exiting.")
    exit(-1)


nn = NN(testNumInput,[(50, 'tanh'), (50, 'tanh'), (testNumOutput, 'softmax')])
nn.train(trainData, trainLabels, 10, 0.01, 0.5);


print( "Model accuracy on training data =",)
acc_train = nn.accuracy(trainData, trainLabels);
print( "%.4f" % acc_train)

print( "Model accuracy on test data     =",)
acc_test = nn.accuracy(testData, testLabels)
print( "%.4f" % acc_test)

print("Done")
