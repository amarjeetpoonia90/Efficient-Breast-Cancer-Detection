import numpy as np
from Evaluation import net_evaluation
from Global_Vars import Global_Vars
from Model_WAM_RCNN import *

def objfun_Segmentation(Soln):
    Images = Global_Vars.Feat
    Tar = Global_Vars.Target
    Model = Global_Vars.Model
    Data_path = Global_Vars.Data_path

    Fitn = np.zeros(Soln.shape[0])
    dimension = len(Soln.shape)
    if dimension == 2:
        for i in range(Soln.shape[0]):
            sol = np.round(Soln[i, :]).astype(np.int16)
            predict = Model_WAM_RCNN_Test(Model, Data_path, Images, sol)
            EVAl = []
            for img in range(len(predict)):
                Eval = net_evaluation(predict[img], Tar[img])
                EVAl.append(Eval)
            mean_EVAl = np.mean(EVAl, axis=0, keepdims=True)
            Fitn[i] = 1 / (mean_EVAl[0, 4] + mean_EVAl[0, 6])  # Dice Coefficient + Accuracy
        return Fitn
    else:
        sol = np.round(Soln).astype(np.int16)
        predict = Model_WAM_RCNN_Test(Model, Data_path, Images, sol)
        EVAl = []
        for img in range(len(predict)):
            Eval = net_evaluation(predict[img], Tar[img])
            EVAl.append(Eval)
        mean_EVAl = np.mean(EVAl, axis=0, keepdims=True)
        Fitn = 1 / (mean_EVAl[0, 4] + mean_EVAl[0, 6])  # Dice Coefficient + Accuracy
        return Fitn