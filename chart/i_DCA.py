import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

def calculate_net_benefit_model(thresh_group, y_pred_score, y_label):
    net_benefit_model = np.array([])
    for thresh in thresh_group:
        y_pred_label = y_pred_score > thresh
        tn, fp, fn, tp = confusion_matrix(y_label, y_pred_label).ravel()
        n = len(y_label)
        net_benefit = (tp / n) - (fp / n) * (thresh / (1 - thresh))
        net_benefit_model = np.append(net_benefit_model, net_benefit)
    return net_benefit_model


def calculate_net_benefit_all(thresh_group, y_label):
    net_benefit_all = np.array([])
    tn, fp, fn, tp = confusion_matrix(y_label, y_label).ravel()
    total = tp + tn
    for thresh in thresh_group:
        net_benefit = (tp / total) - (tn / total) * (thresh / (1 - thresh))
        net_benefit_all = np.append(net_benefit_all, net_benefit)
    return net_benefit_all

def plot_DCA(ax, thresh_group, net_benefit_model_LGB,net_benefit_model_XGB,net_benefit_model_RF,net_benefit_model_LR,net_benefit_model_CNN,net_benefit_model_FNN,net_benefit_model_LR_BMI, net_benefit_all):
    #Plot
    #['red', 'orange', 'black', 'green', 'blue', 'purple', 'pink']
    ax.plot(thresh_group, net_benefit_model_LGB, color = 'red', label = 'LGB')
    ax.plot(thresh_group, net_benefit_model_XGB, color='orange', label='XGB')
    ax.plot(thresh_group, net_benefit_model_RF, color = 'black', label = 'RF')
    ax.plot(thresh_group, net_benefit_model_FNN, color='green', label='FNN')
    ax.plot(thresh_group, net_benefit_model_CNN, color = 'blue', label = 'CNN')
    ax.plot(thresh_group, net_benefit_model_LR, color='purple', label='LOG')
    ax.plot(thresh_group, net_benefit_model_LR_BMI, color='pink', label='LOG2')

    ax.plot(thresh_group, net_benefit_all, color = 'black',label = 'Treat all')
    ax.plot((0, 1), (0, 0), color = 'black', linestyle = ':', label = 'Treat none')

    #Fill，显示出模型较于treat all和treat none好的部分
    y2 = np.maximum(net_benefit_all, 0)
    y1 = np.maximum(net_benefit_model_XGB, y2)
    # ax.fill_between(thresh_group, y1, y2, color = 'crimson', alpha = 0.2)

    #Figure Configuration， 美化一下细节
    ax.set_xlim(0,1)
    ax.set_ylim(net_benefit_model_XGB.min() - 0.15, net_benefit_model_XGB.max() + 0.15)#adjustify the y axis limitation
    ax.set_xlabel(
        xlabel = 'Threshold Probability',
        fontdict= {'family': 'Times New Roman', 'fontsize': 15}
        )
    ax.set_ylabel(
        ylabel = 'Net Benefit',
        fontdict= {'family': 'Times New Roman', 'fontsize': 15}
        )
    ax.grid('major')
    ax.spines['right'].set_color((0.8, 0.8, 0.8))
    ax.spines['top'].set_color((0.8, 0.8, 0.8))
    ax.legend(loc = 'upper right')


    return ax


if __name__ == '__main__':
    #构造一个分类效果不是很好的模型
    df = pd.read_csv('train_proba_all.csv')


    y_pred_score = df['y_lgbm_predict']
    y_label = df['y_true']
    thresh_group = np.arange(0,1,0.01)
    net_benefit_model_LGB = calculate_net_benefit_model(thresh_group, y_pred_score, y_label)
    net_benefit_all = calculate_net_benefit_all(thresh_group, y_label)


    y_pred_score = df['y_xgb_train']
    y_label = df['y_true']
    thresh_group = np.arange(0,1,0.01)
    net_benefit_model_XGB = calculate_net_benefit_model(thresh_group, y_pred_score, y_label)


    y_pred_score = df['y_rf_train']
    y_label = df['y_true']
    thresh_group = np.arange(0,1,0.01)
    net_benefit_model_RF = calculate_net_benefit_model(thresh_group, y_pred_score, y_label)

    y_pred_score = df['y_dnn_train']
    y_label = df['y_true']
    thresh_group = np.arange(0,1,0.01)
    net_benefit_model_FNN = calculate_net_benefit_model(thresh_group, y_pred_score, y_label)

    y_pred_score = df['y_cnn_train']
    y_label = df['y_true']
    thresh_group = np.arange(0,1,0.01)
    net_benefit_model_CNN = calculate_net_benefit_model(thresh_group, y_pred_score, y_label)

    y_pred_score = df['y_lr_train']
    y_label = df['y_true']
    thresh_group = np.arange(0,1,0.01)
    net_benefit_model_LR = calculate_net_benefit_model(thresh_group, y_pred_score, y_label)

    y_pred_score = df['y_lr_BMI_predict']
    y_label = df['y_true']
    thresh_group = np.arange(0,1,0.01)
    net_benefit_model_LR_BMI = calculate_net_benefit_model(thresh_group, y_pred_score, y_label)


    fig, ax = plt.subplots()
    ax = plot_DCA(ax, thresh_group, net_benefit_model_LGB,net_benefit_model_XGB,net_benefit_model_RF,net_benefit_model_LR,net_benefit_model_CNN,net_benefit_model_FNN,net_benefit_model_LR_BMI, net_benefit_all)
    plt.title('train DCA', fontsize=15)
    fig.savefig('train_DCA.png', dpi = 300)
    plt.show()

    df = pd.read_csv('train_proba_all.csv')


    y_pred_score = df['y_lgbm_predict']
    y_label = df['y_true']
    thresh_group = np.arange(0,1,0.01)
    net_benefit_model_LGB = calculate_net_benefit_model(thresh_group, y_pred_score, y_label)
    net_benefit_all = calculate_net_benefit_all(thresh_group, y_label)


    y_pred_score = df['y_xgb_train']
    y_label = df['y_true']
    thresh_group = np.arange(0,1,0.01)
    net_benefit_model_XGB = calculate_net_benefit_model(thresh_group, y_pred_score, y_label)


    y_pred_score = df['y_rf_train']
    y_label = df['y_true']
    thresh_group = np.arange(0,1,0.01)
    net_benefit_model_RF = calculate_net_benefit_model(thresh_group, y_pred_score, y_label)

    y_pred_score = df['y_dnn_train']
    y_label = df['y_true']
    thresh_group = np.arange(0,1,0.01)
    net_benefit_model_FNN = calculate_net_benefit_model(thresh_group, y_pred_score, y_label)

    y_pred_score = df['y_cnn_train']
    y_label = df['y_true']
    thresh_group = np.arange(0,1,0.01)
    net_benefit_model_CNN = calculate_net_benefit_model(thresh_group, y_pred_score, y_label)

    y_pred_score = df['y_lr_train']
    y_label = df['y_true']
    thresh_group = np.arange(0,1,0.01)
    net_benefit_model_LR = calculate_net_benefit_model(thresh_group, y_pred_score, y_label)

    y_pred_score = df['y_lr_BMI_predict']
    y_label = df['y_true']
    thresh_group = np.arange(0,1,0.01)
    net_benefit_model_LR_BMI = calculate_net_benefit_model(thresh_group, y_pred_score, y_label)


    fig, ax = plt.subplots()
    ax = plot_DCA(ax, thresh_group, net_benefit_model_LGB,net_benefit_model_XGB,net_benefit_model_RF,net_benefit_model_LR,net_benefit_model_CNN,net_benefit_model_FNN,net_benefit_model_LR_BMI, net_benefit_all)
    plt.title('Train DCA', fontsize=15)
    fig.savefig('train_DCA.png', dpi = 300)
    plt.show()


    df = pd.read_csv('val_proba_all.csv')


    y_pred_score = df['y_lgbm_predict']
    y_label = df['y_true']
    thresh_group = np.arange(0,1,0.01)
    net_benefit_model_LGB = calculate_net_benefit_model(thresh_group, y_pred_score, y_label)
    net_benefit_all = calculate_net_benefit_all(thresh_group, y_label)


    y_pred_score = df['y_xgb_train']
    y_label = df['y_true']
    thresh_group = np.arange(0,1,0.01)
    net_benefit_model_XGB = calculate_net_benefit_model(thresh_group, y_pred_score, y_label)


    y_pred_score = df['y_rf_train']
    y_label = df['y_true']
    thresh_group = np.arange(0,1,0.01)
    net_benefit_model_RF = calculate_net_benefit_model(thresh_group, y_pred_score, y_label)

    y_pred_score = df['y_dnn_train']
    y_label = df['y_true']
    thresh_group = np.arange(0,1,0.01)
    net_benefit_model_FNN = calculate_net_benefit_model(thresh_group, y_pred_score, y_label)

    y_pred_score = df['y_cnn_train']
    y_label = df['y_true']
    thresh_group = np.arange(0,1,0.01)
    net_benefit_model_CNN = calculate_net_benefit_model(thresh_group, y_pred_score, y_label)

    y_pred_score = df['y_lr_train']
    y_label = df['y_true']
    thresh_group = np.arange(0,1,0.01)
    net_benefit_model_LR = calculate_net_benefit_model(thresh_group, y_pred_score, y_label)

    y_pred_score = df['y_lr_BMI_predict']
    y_label = df['y_true']
    thresh_group = np.arange(0,1,0.01)
    net_benefit_model_LR_BMI = calculate_net_benefit_model(thresh_group, y_pred_score, y_label)


    fig, ax = plt.subplots()
    ax = plot_DCA(ax, thresh_group, net_benefit_model_LGB,net_benefit_model_XGB,net_benefit_model_RF,net_benefit_model_LR,net_benefit_model_CNN,net_benefit_model_FNN,net_benefit_model_LR_BMI, net_benefit_all)
    plt.title('Validation DCA', fontsize=15)
    fig.savefig('val_DCA.png', dpi = 300)
    plt.show()


    df = pd.read_csv('test_proba_all.csv')


    y_pred_score = df['y_lgbm_predict']
    y_label = df['y_true']
    thresh_group = np.arange(0,1,0.01)
    net_benefit_model_LGB = calculate_net_benefit_model(thresh_group, y_pred_score, y_label)
    net_benefit_all = calculate_net_benefit_all(thresh_group, y_label)


    y_pred_score = df['y_xgb_train']
    y_label = df['y_true']
    thresh_group = np.arange(0,1,0.01)
    net_benefit_model_XGB = calculate_net_benefit_model(thresh_group, y_pred_score, y_label)


    y_pred_score = df['y_rf_train']
    y_label = df['y_true']
    thresh_group = np.arange(0,1,0.01)
    net_benefit_model_RF = calculate_net_benefit_model(thresh_group, y_pred_score, y_label)

    y_pred_score = df['y_dnn_train']
    y_label = df['y_true']
    thresh_group = np.arange(0,1,0.01)
    net_benefit_model_FNN = calculate_net_benefit_model(thresh_group, y_pred_score, y_label)

    y_pred_score = df['y_cnn_train']
    y_label = df['y_true']
    thresh_group = np.arange(0,1,0.01)
    net_benefit_model_CNN = calculate_net_benefit_model(thresh_group, y_pred_score, y_label)

    y_pred_score = df['y_lr_train']
    y_label = df['y_true']
    thresh_group = np.arange(0,1,0.01)
    net_benefit_model_LR = calculate_net_benefit_model(thresh_group, y_pred_score, y_label)

    y_pred_score = df['y_lr_BMI_predict']
    y_label = df['y_true']
    thresh_group = np.arange(0,1,0.01)
    net_benefit_model_LR_BMI = calculate_net_benefit_model(thresh_group, y_pred_score, y_label)


    fig, ax = plt.subplots()
    ax = plot_DCA(ax, thresh_group, net_benefit_model_LGB,net_benefit_model_XGB,net_benefit_model_RF,net_benefit_model_LR,net_benefit_model_CNN,net_benefit_model_FNN,net_benefit_model_LR_BMI, net_benefit_all)
    plt.title('Test DCA', fontsize=15)
    fig.savefig('test_DCA.png', dpi = 300)
    plt.show()


    df = pd.read_csv('shhs_proba_all.csv')


    y_pred_score = df['y_lgbm_predict']
    y_label = df['y_true']
    thresh_group = np.arange(0,1,0.01)
    net_benefit_model_LGB = calculate_net_benefit_model(thresh_group, y_pred_score, y_label)
    net_benefit_all = calculate_net_benefit_all(thresh_group, y_label)


    y_pred_score = df['y_xgb_train']
    y_label = df['y_true']
    thresh_group = np.arange(0,1,0.01)
    net_benefit_model_XGB = calculate_net_benefit_model(thresh_group, y_pred_score, y_label)


    y_pred_score = df['y_rf_train']
    y_label = df['y_true']
    thresh_group = np.arange(0,1,0.01)
    net_benefit_model_RF = calculate_net_benefit_model(thresh_group, y_pred_score, y_label)

    y_pred_score = df['y_dnn_train']
    y_label = df['y_true']
    thresh_group = np.arange(0,1,0.01)
    net_benefit_model_FNN = calculate_net_benefit_model(thresh_group, y_pred_score, y_label)

    y_pred_score = df['y_cnn_train']
    y_label = df['y_true']
    thresh_group = np.arange(0,1,0.01)
    net_benefit_model_CNN = calculate_net_benefit_model(thresh_group, y_pred_score, y_label)

    y_pred_score = df['y_lr_train']
    y_label = df['y_true']
    thresh_group = np.arange(0,1,0.01)
    net_benefit_model_LR = calculate_net_benefit_model(thresh_group, y_pred_score, y_label)

    y_pred_score = df['y_lr_BMI_predict']
    y_label = df['y_true']
    thresh_group = np.arange(0,1,0.01)
    net_benefit_model_LR_BMI = calculate_net_benefit_model(thresh_group, y_pred_score, y_label)


    fig, ax = plt.subplots()
    ax = plot_DCA(ax, thresh_group, net_benefit_model_LGB,net_benefit_model_XGB,net_benefit_model_RF,net_benefit_model_LR,net_benefit_model_CNN,net_benefit_model_FNN,net_benefit_model_LR_BMI, net_benefit_all)
    plt.title('Shhs DCA', fontsize=15)
    fig.savefig('shhs_DCA.png', dpi = 300)
    plt.show()



