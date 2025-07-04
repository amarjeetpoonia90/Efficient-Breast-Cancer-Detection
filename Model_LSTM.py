import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from Evaluation import evaluation


def Model_LSTM(train_data, train_target, test_data, test_target, BS=None, sol=None):
    if BS is None:
        BS = 4
    if sol is None:
        sol = [5, 5, 5]

    print('Model LSTM')
    out, model = LSTM_train_1(train_data, train_target, test_data, test_target, BS, sol)
    pred = np.asarray(out)

    Eval = evaluation(pred, test_target)
    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0
    return Eval, pred


def LSTM_train_1(trainX, trainY, testX, testy, Batchsize, sol):
    IMG_SIZE = [1, 100]
    Train_Temp = np.zeros((trainX.shape[0], IMG_SIZE[0], IMG_SIZE[1]))
    for i in range(trainX.shape[0]):
        Train_Temp[i, :] = np.resize(trainX[i], (IMG_SIZE[0], IMG_SIZE[1]))
    Train_X = Train_Temp.reshape(Train_Temp.shape[0], IMG_SIZE[0], IMG_SIZE[1])

    Test_Temp = np.zeros((testy.shape[0], IMG_SIZE[0], IMG_SIZE[1]))
    for i in range(testy.shape[0]):
        Test_Temp[i, :] = np.resize(testy[i], (IMG_SIZE[0], IMG_SIZE[1]))
    Test_X = Test_Temp.reshape(Test_Temp.shape[0], IMG_SIZE[0], IMG_SIZE[1])

    model = Sequential()
    classes = trainY.shape[-1]
    model.add(LSTM(5, input_shape=(Train_X.shape[1], Train_X.shape[-1])))  # hidden neuron count(5 - 255)
    model.add(Dense(50, activation="relu"))
    model.add(Dense(classes, activation="relu"))  # activation="relu"
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(Train_X, trainY, epochs=50, steps_per_epoch=10, batch_size=Batchsize, verbose=2, validation_data=(Test_X, testy))
    testPredict = model.predict(Test_X)
    return testPredict, model

