import numpy as np
from sklearn.metrics import roc_curve
from itertools import cycle
from prettytable import PrettyTable
import matplotlib.pyplot as plt
import cv2 as cv
from matplotlib import pylab


def Statistical(data):
    Min = np.min(data)
    Max = np.max(data)
    Mean = np.mean(data)
    Median = np.median(data)
    Std = np.std(data)
    return np.asarray([Min, Max, Mean, Median, Std])


no_of_dataset = 2


def plot_Con_results():
    Fitness = np.load('Fitness.npy', allow_pickle=True)
    Algorithm = ['TERMS', 'AOA-WAM-RCNN', 'TFMOA-WAM-RCNN', 'SCO-WAM-RCNN', 'FFO-WAM-RCNN', 'EFFO-WAM-RCNN']
    Terms = ['BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD']
    for n in range(Fitness.shape[0]):
        Conv_Graph = np.zeros((5, 5))
        for j in range(5):
            Conv_Graph[j, :] = Statistical(Fitness[n, j, :])
        Table = PrettyTable()
        Table.add_column(Algorithm[0], Terms)
        for j in range(len(Algorithm) - 1):
            Table.add_column(Algorithm[j + 1], Conv_Graph[j, :])
        print('------------------------------ Statistical Report Dataset', n + 1,
              '------------------------------')
        print(Table)

        length = np.arange(Fitness.shape[2])
        Conv_Graph = Fitness[n]

        plt.plot(length, Conv_Graph[0, :], color='r', linewidth=2, label='AOA-WAM-RCNN')
        plt.plot(length, Conv_Graph[1, :], color='g', linewidth=2, label='TFMOA-WAM-RCNN')
        plt.plot(length, Conv_Graph[2, :], color='b', linewidth=2, label='SCO-WAM-RCNN')
        plt.plot(length, Conv_Graph[3, :], color='m', linewidth=2, label='FFO-WAM-RCNN')
        plt.plot(length, Conv_Graph[4, :], color='k', linewidth=2, label='EFFO-WAM-RCNN')
        plt.xlabel('Iteration')
        plt.ylabel('Cost Function')
        plt.legend(loc=1)
        plt.savefig("./Results/Convergence_%s.png" % (n + 1))
        fig = pylab.gcf()
        fig.canvas.manager.set_window_title('Convergence graph of dataset' + str(n+1))
        plt.show()


def ROC_curve():
    lw = 2

    cls = ['ResNet', 'Inception', 'MobileNet', 'DensNet', 'ViT-MobLSTM']
    for n in range(no_of_dataset):
        Actual = np.load('Classification_target_' + str(n + 1) + '.npy', allow_pickle=True).astype('int')

        colors = cycle(
            ["#fe2f4a", "#0165fc", "#ffff14", "lime", "black"])
        for i, color in zip(range(len(cls)), colors):
            Predicted = np.load('Y_Score_' + str(n + 1) + '.npy', allow_pickle=True)[i]
            false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(Actual.ravel(), Predicted.ravel())
            plt.plot(
                false_positive_rate1,
                true_positive_rate1,
                color=color,
                lw=lw,
                label=cls[i],
            )
        plt.plot([0, 1], [0, 1], "k--", lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        path = "./Results/ROC_%s.png" % (n + 1)
        plt.savefig(path)
        fig = pylab.gcf()
        fig.canvas.manager.set_window_title('ROC curve of Dataset ' + str(n+1))
        plt.show()


def Plot_Batch_Size():
    eval = np.load('Eval_ALL.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'FOR', 'NPV', 'FDR', 'F1_score',
             'MCC',
             'pt',
             'ba', 'fm', 'bm', 'mk', 'PLHR', 'lrminus', 'dor', 'prevalence', 'TS']

    Table_Term = [0, 2, 3, 5, 10, 11, 13, 14, 15, 16, 20]
    positive_metrices = [0, 1, 2, 3, 7, 9, 10]
    negative_metrices = [4, 5, 6, 8]
    Batchsize = [4, 8, 16, 32, 48, 64]
    Graph_Term = np.arange(len(Terms))

    Classifier = ['TERMS', 'ResNet', 'Inception', 'MobileNet', 'DensNet', 'ViT-MobLSTM']
    for i in range(eval.shape[0]):
        for k in range(eval.shape[1]):
            if k == 4:
                pass
            else:
                value = eval[i, k, :, 4:]
                Table = PrettyTable()
                Table.add_column(Classifier[0], (np.asarray(Terms))[np.asarray(Table_Term)])
                for j in range(len(Classifier) - 1):
                    Table.add_column(Classifier[j + 1], value[j, Table_Term])
                print('-------------------------------------------------- ', str(Batchsize[k]), ' batch size ',
                      'Classifier Comparison of Dataset', i + 1,
                      '--------------------------------------------------')
                print(Table)

    for i in range(eval.shape[0]):
        Graph = np.zeros((eval.shape[2], eval.shape[3] - 4))
        for l in range(eval.shape[2]):
            for j in range(len(Graph_Term)):
                Graph[l, j] = eval[i, 4, l, Graph_Term[j] + 4]

        # Positive measures
        length = np.arange(len(positive_metrices))
        fig = plt.figure()
        ax = fig.add_axes([0.15, 0.1, 0.7, 0.8])
        Alg_Val = Graph[:5, positive_metrices]

        ax.plot(length, Alg_Val[0, :], color='#010fcc', linewidth=3, marker='>', markerfacecolor='red',
                markersize=12, label='ResNet')
        ax.plot(length, Alg_Val[1, :], color='#08ff08', linewidth=3, marker='>', markerfacecolor='green',
                markersize=12, label='Inception')
        ax.plot(length, Alg_Val[2, :], color='#fe420f', linewidth=3, marker='>', markerfacecolor='cyan',
                markersize=12, label='MobileNet')
        ax.plot(length, Alg_Val[3, :], color='#f504c9', linewidth=3, marker='>', markerfacecolor='#fdff38',
                markersize=12, label='DensNet')
        ax.plot(length, Alg_Val[4, :], color='k', linewidth=3, marker='>', markerfacecolor='w', markersize=12,
                label='ViT-MobLSTM')
        ax.fill_between(length, Alg_Val[0, :], Alg_Val[2, :], color='#acc2d9', alpha=.5)
        ax.fill_between(length, Alg_Val[0, :], Alg_Val[1, :], color='#c48efd', alpha=.5)
        ax.fill_between(length, Alg_Val[1, :], Alg_Val[3, :], color='#ff6f52', alpha=.5)
        ax.fill_between(length, Alg_Val[3, :], Alg_Val[4, :], color='#b2fba5', alpha=.5)
        plt.xticks(length, ('Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'NPV', 'F1_score', 'MCC'),
                   rotation=10, fontsize=9, fontname="Arial", fontweight='bold')
        plt.ylabel('Positive Measures (%)', fontname="Arial", fontsize=12, fontweight='bold', color='#35530a')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.14), ncol=3, fancybox=True, shadow=False)
        path = "./Results/Batch_Size_Positive_Dataset_%s_line.png" % (i + 1)
        plt.savefig(path)
        fig = pylab.gcf()
        fig.canvas.manager.set_window_title('Batch_Size vs Positive Measures')
        plt.show()

        # Negative measures
        length = np.arange(len(negative_metrices))
        fig = plt.figure()
        ax = fig.add_axes([0.15, 0.1, 0.7, 0.8])
        Alg_Val = Graph[:5, negative_metrices]

        ax.plot(length, Alg_Val[0, :], color='#010fcc', linewidth=3, marker='>', markerfacecolor='red',  # 98F5FF
                markersize=12, label='ResNet')
        ax.plot(length, Alg_Val[1, :], color='#08ff08', linewidth=3, marker='>', markerfacecolor='green',  # 7FFF00
                markersize=12, label='Inception')
        ax.plot(length, Alg_Val[2, :], color='#fe420f', linewidth=3, marker='>', markerfacecolor='cyan',  # C1FFC1
                markersize=12, label='MobileNet')
        ax.plot(length, Alg_Val[3, :], color='#f504c9', linewidth=3, marker='>', markerfacecolor='#fdff38',
                markersize=12, label='DensNet')
        ax.plot(length, Alg_Val[4, :], color='k', linewidth=3, marker='>', markerfacecolor='w', markersize=12,
                label='ViT-MobLSTM')
        ax.fill_between(length, Alg_Val[0, :], Alg_Val[2, :], color='#acc2d9', alpha=.5)  # ff8400
        ax.fill_between(length, Alg_Val[0, :], Alg_Val[1, :], color='#c48efd', alpha=.5)  # 19abff
        ax.fill_between(length, Alg_Val[1, :], Alg_Val[3, :], color='#ff6f52', alpha=.5)  # 00f7ff
        ax.fill_between(length, Alg_Val[3, :], Alg_Val[4, :], color='#b2fba5', alpha=.5)  # ecfc5b
        plt.xticks(length, ('FPR', 'FNR', 'FOR', 'FDR'),
                   rotation=10, fontsize=9, fontname="Arial", fontweight='bold')
        plt.ylabel('Negative Measures (%)', fontname="Arial", fontsize=12, fontweight='bold', color='#35530a')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.14), ncol=3, fancybox=True, shadow=False)
        path = "./Results/Batch_Size_Negative_Dataset_%s_line.png" % (i + 1)
        plt.savefig(path)
        fig = pylab.gcf()
        fig.canvas.manager.set_window_title('Batch_Size vs Negative Measures')
        plt.show()


def plot_results_Seg():
    Eval_all = np.load('Eval_all_seg.npy', allow_pickle=True)
    Terms = ['Dice Coefficient', 'Jaccard', 'Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV',
             'FDR', 'F1-Score', 'MCC']
    for n in range(Eval_all.shape[0]):
        value_all = Eval_all[n, :]
        stats = np.zeros((value_all[0].shape[1] - 4, value_all.shape[0] + 4, 5))
        for i in range(4, value_all[0].shape[1] - 9):
            for j in range(value_all.shape[0] + 4):
                if j < value_all.shape[0]:
                    stats[i, j, 0] = np.max(value_all[j][:, i]) * 100
                    stats[i, j, 1] = np.min(value_all[j][:, i]) * 100
                    stats[i, j, 2] = np.mean(value_all[j][:, i]) * 100
                    stats[i, j, 3] = np.median(value_all[j][:, i]) * 100
                    stats[i, j, 4] = np.std(value_all[j][:, i]) * 100

            X = np.arange(stats.shape[2])
            fig = plt.figure()
            ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])

            ax.bar(X + 0.00, stats[i, 0, :], color='#f075e6', edgecolor='w', width=0.10, label="AOA-WAM-RCNN")  # r
            ax.bar(X + 0.10, stats[i, 1, :], color='#0cff0c', edgecolor='w', width=0.10, label="TFMOA-WAM-RCNN")  # g
            ax.bar(X + 0.20, stats[i, 2, :], color='#0165fc', edgecolor='w', width=0.10, label="SCO-WAM-RCNN")  # b
            ax.bar(X + 0.30, stats[i, 3, :], color='#fd411e', edgecolor='w', width=0.10, label="FFO-WAM-RCNN")  # m
            ax.bar(X + 0.40, stats[i, 4, :], color='k', edgecolor='w', width=0.10, label="EFFO-WAM-RCNN")  # k
            plt.xticks(X + 0.20, ('BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD'))
            plt.xlabel('Statisticsal Analysis')
            plt.ylabel(Terms[i - 4])
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=True)
            path = "./Results/Dataset_%s_Seg_%s_alg.png" % (n + 1, Terms[i - 4])
            plt.savefig(path)
            fig = pylab.gcf()
            fig.canvas.manager.set_window_title('Statisticsal Analysis vs ' + Terms[i - 4])
            plt.show()

            X = np.arange(stats.shape[2])
            fig = plt.figure()
            ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])

            ax.bar(X + 0.00, stats[i, 5, :], color='#ff028d', edgecolor='k', width=0.10, label="UNet")
            ax.bar(X + 0.10, stats[i, 6, :], color='#0cff0c', edgecolor='k', width=0.10, label="ResUNet")
            ax.bar(X + 0.20, stats[i, 7, :], color='#0165fc', edgecolor='k', width=0.10, label="TransUNet")
            ax.bar(X + 0.30, stats[i, 8, :], color='#fd411e', edgecolor='k', width=0.10, label="WAM-RCNN")
            ax.bar(X + 0.40, stats[i, 4, :], color='k', edgecolor='k', width=0.10, label="EFFO-WAM-RCNN")
            plt.xticks(X + 0.20, ('BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD'))
            plt.xlabel('Statisticsal Analysis')
            plt.ylabel(Terms[i - 4])
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=True)
            path = "./Results/Dataset_%s_Seg_%s_mtd.png" % (n + 1, Terms[i - 4])
            plt.savefig(path)
            fig = pylab.gcf()
            fig.canvas.manager.set_window_title('Statisticsal Analysis vs ' + Terms[i - 4])
            plt.show()


def Image_segment_comparision():
    for n in range(no_of_dataset):
        Original = np.load('Images_' + str(n + 1) + '.npy', allow_pickle=True)
        segmented = np.load('Seg_img_' + str(n + 1) + '.npy', allow_pickle=True)
        Ground_truth = np.load('Ground_Truth_' + str(n + 1) + '.npy', allow_pickle=True)
        if n == 0:
            Image = [30, 31, 32, 261, 253, 252]
        elif n == 1:
            Image = [2, 4, 10, 12, 66, 121, 136]

        for i in range(5):
            Orig = Original[Image[i]]
            Seg_1 = segmented[Image[i]]
            GT = Ground_truth[Image[i]]
            for j in range(1):
                Orig_1 = Seg_1[j]
                Orig_2 = Seg_1[j + 1]
                Orig_3 = Seg_1[j + 2]
                Orig_4 = Seg_1[j + 3]
                Orig_5 = Seg_1[j + 4]
                plt.suptitle('Segmented Images from Dataset ', fontsize=20)

                plt.subplot(3, 3, 1).axis('off')
                plt.imshow(GT)
                plt.title('Ground Truth', fontsize=10)

                plt.subplot(3, 3, 2).axis('off')
                plt.imshow(Orig_1)
                plt.title('Unet', fontsize=10)

                plt.subplot(3, 3, 3).axis('off')
                plt.imshow(Orig_2)
                plt.title('ResUNet', fontsize=10)

                plt.subplot(3, 3, 5).axis('off')
                plt.imshow(Orig)
                plt.title('Original', fontsize=10)

                plt.subplot(3, 3, 7).axis('off')
                plt.imshow(Orig_3)
                plt.title('TransUnet ', fontsize=10)

                plt.subplot(3, 3, 8).axis('off')
                plt.imshow(Orig_4)
                plt.title('WAM-RCNN', fontsize=10)

                plt.subplot(3, 3, 9).axis('off')
                plt.imshow(Orig_5)
                plt.title('EFFO-WAM-RCNN', fontsize=10)

                path = "./Results/Image_Results/Dataset_%s_image_%s.png" % (n + 1, i + 1)
                plt.savefig(path)
                plt.show()

                cv.imwrite('./Results/Image_Results/Dataset_' + str(n + 1) + 'Orig_image_' + str(i + 1) + '.png', Orig)
                cv.imwrite('./Results/Image_Results/Dataset_' + str(n + 1) + 'Ground_Truth_' + str(i + 1) + '.png', GT)
                cv.imwrite('./Results/Image_Results/Dataset_' + str(n + 1) + 'segm_Unet_' + str(i + 1) + '.png', Orig_1)
                cv.imwrite('./Results/Image_Results/Dataset_' + str(n + 1) + 'segm_ResUNet_' + str(i + 1) + '.png',
                           Orig_2)
                cv.imwrite('./Results/Image_Results/Dataset_' + str(n + 1) + 'segm_TransUnet_' + str(i + 1) + '.png',
                           Orig_3)
                cv.imwrite('./Results/Image_Results/Dataset_' + str(n + 1) + 'segm_WAM_RCNN_' + str(i + 1) + '.png',
                           Orig_4)
                cv.imwrite(
                    './Results/Image_Results/Dataset_' + str(n + 1) + 'segm_EFFO_WAM_RCNN_' + str(i + 1) + '.png',
                    Orig_5)


if __name__ == '__main__':
    plot_Con_results()
    ROC_curve()
    Plot_Batch_Size()
    plot_results_Seg()
    Image_segment_comparision()
