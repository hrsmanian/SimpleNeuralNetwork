import numpy as np;
from NN import NN

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

inputFileName = "mnist_test_encoded.csv";
testData = ReadData(inputFileName);
nn = NN(testData[1],[(50, 'tanh'), (50, 'tanh'), (testData[2], 'softmax')])
trainData = ReadData("mnist_train_encoded.csv");

nn.train(trainData[3], trainData[4], 10, 0.01, 0.5);


print( "Model accuracy on training data =",)
acc_train = nn.accuracy(trainData[3], trainData[4]);
print( "%.4f" % acc_train)

print( "Model accuracy on test data     =",)
acc_test = nn.accuracy(testData[3], testData[4])
print( "%.4f" % acc_test)

print("Done")
