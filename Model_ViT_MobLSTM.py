import numpy as np
from sklearn.model_selection import train_test_split
from Model_LSTM import Model_LSTM
from Model_VIT_MobileNet import Model_Vit_MobileNet


def Model_Vit_MobLSTM(train_data, train_labels, test_data, test_labels, BS=None):
    if BS is None:
        BS = 4

    Feature = Model_Vit_MobileNet(train_data, train_labels, test_data, test_labels, BS=BS)
    y = np.concatenate((train_labels, test_labels), axis=0)
    X_trainFeat, X_testFeat, y_trainFeat, y_testFeat = train_test_split(Feature, y, test_size=0.2, random_state=42)
    EVAL, pred = Model_LSTM(X_trainFeat, y_trainFeat, X_testFeat, y_testFeat, BS=BS)
    return EVAL, pred

