import numpy as np
import os
import pandas as pd
from numpy import matlib
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from AOA import AOA
from FFO import FFO
from Global_Vars import Global_Vars
from Model_DenseNet import Model_DenseNet
from Model_Inception import Model_Inception
from Model_MobileNet import Model_MobileNet
from Model_Resnet import Model_RESNET
from Model_Trans_Unet import Model_Trans_Unet
from Model_UNET import Model_Unet
from Model_Unet3Plus import Model_Unet3plus
from Model_ViT_MobLSTM import Model_Vit_MobLSTM
from Model_WAM_RCNN import *
from Objective_Function import objfun_Segmentation
from Proposed import PROPOSED
from SCO import SCO
from TFMOA import TFMOA
from Plot_results import *

no_of_dataset = 2


# Read Dataset 1
def Read_Dataset_1():
    original = []
    Directory = './Dataset/Dataset1/all-mias/'
    out_folder = os.listdir(Directory)
    for i in range(len(out_folder)):
        filename = Directory + out_folder[i]
        if '.txt' in filename:
            pass
        elif '.pgm' in filename:
            Data = cv.imread(filename)
            Data_img = cv.resize(Data, (256, 256), interpolation=cv.INTER_NEAREST)
            original.append(Data_img)
        else:
            Excel = pd.read_excel(filename)
            Tar = np.asarray(Excel['G CLA'])
            ind = np.where(Tar == 'NORM')
            Detect_tar = np.ones((len(Tar))).astype('int')
            Detect_tar[ind] = 0

            label_encoder = LabelEncoder()
            Tar_encoded = label_encoder.fit_transform(Tar)
            class_tar = to_categorical(Tar_encoded, dtype="uint8")

    return np.asarray(original), class_tar


# Read Dataset 2
def Read_Dataset_2():
    Images = []
    TrainDir = './Dataset/Dataset2/mammography_images/Training_set.csv'
    TestDir = './Dataset/Dataset2/mammography_images/sample_submission.csv'

    Train = pd.read_csv(TrainDir)
    Test = pd.read_csv(TestDir)
    Train = np.asarray(Train)
    Test = np.asarray(Test)
    Train_data = []
    for i in range(len(Train)):
        naming = './Dataset/Dataset2/mammography_images/train/' + Train[i][0]
        Train_data_tar = np.append(naming, Train[i, 1])
        Train_data.append(Train_data_tar)

    Test_data = []
    for j in range(len(Test)):
        naming = './Dataset/Dataset2/mammography_images/test/' + Test[j][0]
        Test_data_tar = np.append(naming, Test[j, 1])
        Test_data.append(Test_data_tar)

    Train_data = np.asarray(Train_data)
    Test_data = np.asarray(Test_data)

    Datas = np.concatenate((Train_data, Test_data), axis=0)

    for k in range(len(Datas)):
        print(k, len(Datas))
        Img = cv.imread(Datas[k, 0])
        Img_Data = cv.resize(Img, (256, 256), interpolation=cv.INTER_NEAREST)
        Images.append(Img_Data)

    Targets = Datas[:, 1]

    Targets = np.asarray(Targets)
    label_encoder = LabelEncoder()
    Tar_encoded = label_encoder.fit_transform(Targets)
    class_tar = to_categorical(Tar_encoded, dtype="uint8")
    Images = np.asarray(Images)
    return Images, class_tar


# Read the Datasets
an = 0
if an == 1:
    Images_1, Target_1 = Read_Dataset_1()
    Images_2, Target_2 = Read_Dataset_2()

    np.save('Images_1.npy', Images_1)
    np.save('Images_2.npy', Images_2)
    np.save('Classification_target_1.npy', Target_1)
    np.save('Classification_target_2.npy', Target_2)

# Segmentation
an = 0
if an == 1:
    for n in range(no_of_dataset):
        if n == 0:
            Data_path = './Images/Original_images/Dataset_1/'
        else:
            Data_path = './Images/Original_images/Dataset_2/'
        Data = np.load('Images_' + str(n + 1) + '.npy', allow_pickle=True)  # Load the Data
        Train_model, Proposed = Model_WAM_RCNN_train(Data_path, Data)
        save_model(Train_model, 'saved_model_' + str(n + 1) + '.pth')

# Optimization For Segmentation
an = 0
if an == 1:
    Best_SOl = []
    FiTness = []
    for n in range(no_of_dataset):
        Data_path = './Images/Original_images/Dataset_' + str(n + 1) + '/'
        Feat = np.load('Segmented_image_' + str(n + 1) + '.npy', allow_pickle=True)
        Target = np.load('Classification_target_' + str(n + 1) + '.npy', allow_pickle=True)
        loaded_model = load_model('saved_model_' + str(n + 1) + '.pth')
        Global_Vars.Feat = Feat
        Global_Vars.Target = Target
        Global_Vars.Data_path = Data_path
        Global_Vars.Model = loaded_model
        Npop = 10
        Chlen = get_hidden_neurons(loaded_model)  # hidden neuron count
        xmin = matlib.repmat(np.asarray([0.01]), Npop, Chlen)
        xmax = matlib.repmat(np.asarray([0.99]), Npop, Chlen)
        fname = objfun_Segmentation
        initsol = np.zeros((Npop, Chlen))
        for p1 in range(initsol.shape[0]):
            for p2 in range(initsol.shape[1]):
                initsol[p1, p2] = np.random.uniform(xmin[p1, p2], xmax[p1, p2])
        Max_iter = 50

        print("AOA...")
        [bestfit1, fitness1, bestsol1, time1] = AOA(initsol, fname, xmin, xmax, Max_iter)  # AOA

        print("TFMOA...")
        [bestfit2, fitness2, bestsol2, time2] = TFMOA(initsol, fname, xmin, xmax, Max_iter)  # TFMOA

        print("SCO...")
        [bestfit3, fitness3, bestsol3, time3] = SCO(initsol, fname, xmin, xmax, Max_iter)  # SCO

        print("FFO...")
        [bestfit4, fitness4, bestsol4, time4] = FFO(initsol, fname, xmin, xmax, Max_iter)  # FFO

        print("PROPOSED...")
        [bestfit5, fitness5, bestsol5, time5] = PROPOSED(initsol, fname, xmin, xmax, Max_iter)  # PROPOSED

        BestSol_CLS = [bestsol1.squeeze(), bestsol2.squeeze(), bestsol3.squeeze(), bestsol4.squeeze(),
                       bestsol5.squeeze()]
        fitness = [fitness1.squeeze(), fitness2.squeeze(), fitness3.squeeze(), fitness4.squeeze(), fitness5.squeeze()]
        Best_SOl.append(BestSol_CLS)
        FiTness.append(fitness)
    np.save('Fitness.npy', np.asarray(FiTness))
    np.save('BestSol.npy', np.asarray(Best_SOl))

# Segmentation
an = 0
if an == 1:
    for n in range(no_of_dataset):
        Data_path = './Images/Original_images/Dataset_' + str(n + 1) + '/'
        Data = np.load('Images_' + str(n + 1) + '.npy', allow_pickle=True)  # Load the Data
        Target = np.load('Ground_Truth_' + str(n + 1) + '.npy', allow_pickle=True)  # Load the ground truth
        BestSol = np.load('BestSol.npy', allow_pickle=True)[n]
        loaded_model = load_model('saved_model_' + str(n + 1) + '.pth')
        Unet = Model_Unet(Data_path)
        Unet3plus = Model_Unet3plus(Data, Target)
        Trans_Unet = Model_Trans_Unet(Data, Target)
        A_SMU = Model_WAM_RCNN_Test(loaded_model, Data_path, Data)
        Proposed = Model_WAM_RCNN_Test(loaded_model, Data_path, Data, BestSol)
        Seg = [Unet, Unet3plus, Trans_Unet, A_SMU, Proposed]
        np.save('Segmented_image_' + str(n + 1) + '.npy', Proposed)
        np.save('Seg_img_' + str(n + 1) + '.npy', Seg)

# Classification by Varying Batch Size
an = 0
if an == 1:
    Evall_All = []
    for n in range(no_of_dataset):
        EVAL = []
        Feat = np.load('Segmented_image_' + str(n + 1) + '.npy', allow_pickle=True)  # Load the Selected features
        Target = np.load('Classification_target_' + str(n + 1) + '.npy', allow_pickle=True)
        Batchsize = [4, 8, 16, 32, 48, 64]
        for Bs in range(len(Batchsize)):
            learnperc = round(Feat.shape[0] * 0.75)
            Train_Data = Feat[:learnperc, :]
            Train_Target = Target[:learnperc]
            Test_Data = Feat[learnperc:, :]
            Test_Target = Target[learnperc:]
            Eval = np.zeros((5, 25))
            Eval[0, :], pred1 = Model_RESNET(Train_Data, Train_Target, Test_Data, Test_Target, BS=Batchsize[Bs])
            Eval[1, :], pred2 = Model_Inception(Train_Data, Train_Target, Test_Data, Test_Target, BS=Batchsize[Bs])
            Eval[2, :], pred3 = Model_MobileNet(Train_Data, Train_Target, Test_Data, Test_Target, BS=Batchsize[Bs])
            Eval[3, :], pred4 = Model_DenseNet(Train_Data, Train_Target, Test_Data, Test_Target, BS=Batchsize[Bs])
            Eval[4, :], pred5 = Model_Vit_MobLSTM(Train_Data, Train_Target, Test_Data, Test_Target, BS=Batchsize[Bs])
            EVAL.append(Eval)
        Evall_All.append(EVAL)
    np.save('Eval_ALL.npy', np.asarray(Evall_All))  # Save the Eval all

plot_Con_results()
ROC_curve()
Plot_Batch_Size()
plot_results_Seg()
Image_segment_comparision()

