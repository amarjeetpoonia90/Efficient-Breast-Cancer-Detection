import numpy as np
from keras.applications import MobileNet
from Evaluation import evaluation
import cv2 as cv


def Model_MobileNet(train_data, train_tar, test_data, test_tar, BS=None, sol=None):
    if BS is None:
        BS = 4
    if sol is None:
        sol = [2, 1, 1]
    model = MobileNet(weights='imagenet')
    IMG_SIZE = [224, 224, 3]

    Train_x = np.zeros((train_data.shape[0], IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2]))
    for i in range(train_data.shape[0]):
        temp = cv.resize(train_data[i], (IMG_SIZE[1] * IMG_SIZE[2], IMG_SIZE[0]))
        Train_x[i] = np.reshape(temp, (IMG_SIZE[0], IMG_SIZE[1], 3))

    Test_X = np.zeros((test_data.shape[0], IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2]))
    for i in range(test_data.shape[0]):
        temp_1 = np.resize(test_data[i], (IMG_SIZE[0] * IMG_SIZE[1], 3))
        Test_X[i] = np.reshape(temp_1, (IMG_SIZE[0], IMG_SIZE[1], 3))

    model.compile(loss='mean_squared_error', optimizer='adam')
    Train_y = np.append(train_tar, (np.zeros((train_tar.shape[0], 999))), axis=1)
    Test_y = np.append(test_tar, (np.zeros((test_tar.shape[0], 999))), axis=1)
    model.fit(Train_x, Train_y[:, :1000], epochs=5, batch_size=BS, validation_data=(Test_X, Test_y[:, :1000]))
    pred = model.predict(Test_X)
    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0
    Eval = evaluation(pred, Test_y[:, :1000])
    return Eval, pred
