import matplotlib.pyplot as plt
import pickle
import os

import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, \
    classification_report, roc_curve, auc

df = pd.read_csv('train_proba_all.csv')

df = df.drop('Unnamed: 0',axis = 1)
print(df)

y_predict = df.drop('y_true',axis = 1)
y_true = df['y_true']

print(df.columns)

def multi_models_roc(names, colors, y_true, y_predict, save=True, dpin=100):
    """
    将多个机器模型的roc图输出到一张图上

    Args:
        names: list, 多个模型的名称
        sampling_methods: list, 多个模型的实例化对象
        save: 选择是否将结果保存（默认为png格式）

    Returns:
        返回图片对象plt
    """
    plt.figure(figsize=(10, 10), dpi=dpin)

    for (name,  colorname,columns) in zip(names, colors,y_predict.columns):

        fpr, tpr, thresholds = roc_curve(y_true,y_predict[columns], pos_label=1)
        print(name)
        print('AUC={:.4f}'.format(auc(fpr, tpr)))

        plt.plot(fpr, tpr, lw=2, label='{} (AUC={:.3f})'.format(name, auc(fpr, tpr)), color=colorname)
        plt.plot([0, 1], [0, 1], '--', lw=2, color='grey')
        plt.axis('square')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xlabel('False Positive Rate', fontsize=20)
        plt.ylabel('True Positive Rate', fontsize=20)
        plt.title('Train ROC Curve', fontsize=25)
        plt.legend(loc='lower right', fontsize=20)

    if save:
        plt.savefig('train_roc.png')

    return plt

name = ['LGB','XGB','RF','FNN','CNN','LOG','LOG2']
colors = ['red','orange','black','green','blue','purple','pink']

multi_models_roc(name,colors,y_true,y_predict)


df = pd.read_csv('val_proba_all.csv')

df = df.drop('Unnamed: 0',axis = 1)
print(df)

y_predict = df.drop('y_true',axis = 1)
y_true = df['y_true']

print("-----val------")
def multi_models_roc(names, colors, y_true, y_predict, save=True, dpin=100):
    """
    将多个机器模型的roc图输出到一张图上

    Args:
        names: list, 多个模型的名称
        sampling_methods: list, 多个模型的实例化对象
        save: 选择是否将结果保存（默认为png格式）

    Returns:
        返回图片对象plt
    """
    plt.figure(figsize=(10, 10), dpi=dpin)

    for (name,  colorname,columns) in zip(names, colors,y_predict.columns):

        fpr, tpr, thresholds = roc_curve(y_true,y_predict[columns], pos_label=1)
        print(name)
        print('AUC={:.4f}'.format(auc(fpr, tpr)))

        plt.plot(fpr, tpr, lw=2, label='{} (AUC={:.3f})'.format(name, auc(fpr, tpr)), color=colorname)
        plt.plot([0, 1], [0, 1], '--', lw=2, color='grey')
        plt.axis('square')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xlabel('False Positive Rate', fontsize=20)
        plt.ylabel('True Positive Rate', fontsize=20)
        plt.title('Validation ROC Curve', fontsize=25)
        plt.legend(loc='lower right', fontsize=20)

    if save:
        plt.savefig('val_roc.png')

    return plt

name = ['LGB','XGB','RF','FNN','CNN','LOG','LOG2']
colors = ['red','orange','black','green','blue','purple','pink']


multi_models_roc(name,colors,y_true,y_predict)

df = pd.read_csv('test_proba_all.csv')

df = df.drop('Unnamed: 0',axis = 1)
print(df)

y_predict = df.drop('y_true',axis = 1)
y_true = df['y_true']

print("-----test------")
def multi_models_roc(names, colors, y_true, y_predict, save=True, dpin=100):
    """
    将多个机器模型的roc图输出到一张图上

    Args:
        names: list, 多个模型的名称
        sampling_methods: list, 多个模型的实例化对象
        save: 选择是否将结果保存（默认为png格式）

    Returns:
        返回图片对象plt
    """
    plt.figure(figsize=(10, 10), dpi=dpin)

    for (name,  colorname,columns) in zip(names, colors,y_predict.columns):

        fpr, tpr, thresholds = roc_curve(y_true,y_predict[columns], pos_label=1)
        print(name)
        print('AUC={:.4f}'.format(auc(fpr, tpr)))

        plt.plot(fpr, tpr, lw=2, label='{} (AUC={:.3f})'.format(name, auc(fpr, tpr)), color=colorname)
        plt.plot([0, 1], [0, 1], '--', lw=2, color='grey')
        plt.axis('square')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xlabel('False Positive Rate', fontsize=20)
        plt.ylabel('True Positive Rate', fontsize=20)
        plt.title('Test ROC Curve', fontsize=25)
        plt.legend(loc='lower right', fontsize=20)

    if save:
        plt.savefig('test_roc.png')

    return plt

name = ['LGB','XGB','RF','FNN','CNN','LOG','LOG2']
colors = ['red','orange','black','green','blue','purple','pink']


multi_models_roc(name,colors,y_true,y_predict)

df = pd.read_csv('shhs_proba_all.csv')

df = df.drop('Unnamed: 0',axis = 1)
print(df)

y_predict = df.drop('y_true',axis = 1)
y_true = df['y_true']

print("-----shhs------")
def multi_models_roc(names, colors, y_true, y_predict, save=True, dpin=100):
    """
    将多个机器模型的roc图输出到一张图上

    Args:
        names: list, 多个模型的名称
        sampling_methods: list, 多个模型的实例化对象
        save: 选择是否将结果保存（默认为png格式）

    Returns:
        返回图片对象plt
    """
    plt.figure(figsize=(10, 10), dpi=dpin)

    for (name,  colorname,columns) in zip(names, colors,y_predict.columns):

        fpr, tpr, thresholds = roc_curve(y_true,y_predict[columns], pos_label=1)
        print(name)
        print('AUC={:.4f}'.format(auc(fpr, tpr)))

        plt.plot(fpr, tpr, lw=2, label='{} (AUC={:.3f})'.format(name, auc(fpr, tpr)), color=colorname)
        plt.plot([0, 1], [0, 1], '--', lw=2, color='grey')
        plt.axis('square')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xlabel('False Positive Rate', fontsize=20)
        plt.ylabel('True Positive Rate', fontsize=20)
        plt.title('Shhs ROC Curve', fontsize=25)
        plt.legend(loc='lower right', fontsize=20)

    if save:
        plt.savefig('shhs_roc.png')

    return plt

name = ['LGB','XGB','RF','FNN','CNN','LOG','LOG2']
colors = ['red','orange','black','green','blue','purple','pink']


multi_models_roc(name,colors,y_true,y_predict)