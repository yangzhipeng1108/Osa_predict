from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, \
    classification_report, roc_curve, auc, roc_auc_score, multilabel_confusion_matrix
import numpy as np
from math import sqrt
import pandas as pd


def roc_auc_ci(y_true, y_score, positive=1):
    AUC = roc_auc_score(y_true, y_score)
    N1 = sum(y_true == positive)
    N2 = sum(y_true != positive)
    Q1 = AUC / (2 - AUC)
    Q2 = 2 * AUC ** 2 / (1 + AUC)
    SE_AUC = sqrt((AUC * (1 - AUC) + (N1 - 1) * (Q1 - AUC ** 2) + (N2 - 1) * (Q2 - AUC ** 2)) / (N1 * N2))
    lower = AUC - 1.96 * SE_AUC
    upper = AUC + 1.96 * SE_AUC
    if lower < 0:
        lower = 0
    if upper > 1:
        upper = 1
    return '%.3f'%AUC + '(' + '%.3f'%lower + ',' + '%.3f'%upper + ')'


def sensitivityCalc(Predictions, Labels):
    MCM = multilabel_confusion_matrix(Labels, Predictions,
                                      sample_weight=None,
                                      labels=None, samplewise=None)
    # MCM此处是 5 * 2 * 2的混淆矩阵（ndarray格式），5表示的是5分类

    # 切片操作，获取每一个类别各自的 tn, fp, tp, fn
    tn_sum = MCM[:, 0, 0]  # True Negative
    fp_sum = MCM[:, 0, 1]  # False Positive

    tp_sum = MCM[:, 1, 1]  # True Positive
    fn_sum = MCM[:, 1, 0]  # False Negative

    # 这里加1e-6，防止 0/0的情况计算得到nan，即tp_sum和fn_sum同时为0的情况
    Condition_negative = tp_sum[1] + fn_sum[0]

    sensitivity = tp_sum[1] / Condition_negative

    return sensitivity


def specificityCalc(Predictions, Labels):
    MCM = multilabel_confusion_matrix(Labels, Predictions,
                                      sample_weight=None,
                                      labels=None, samplewise=None)
    # tn_sum = MCM[:, 0, 0]
    # fp_sum = MCM[:, 0, 1]

    tp_sum = MCM[:, 1, 1]
    fn_sum = MCM[:, 1, 0]

    Condition_negative = tp_sum[0] + fn_sum[1]

    specificity = tp_sum[0] / Condition_negative

    return specificity


def PositiveCalc(Predictions, Labels):
    MCM = multilabel_confusion_matrix(Labels, Predictions,
                                      sample_weight=None,
                                      labels=None, samplewise=None)
    # MCM此处是 5 * 2 * 2的混淆矩阵（ndarray格式），5表示的是5分类

    # 切片操作，获取每一个类别各自的 tn, fp, tp, fn
    # tn_sum = MCM[:, 0, 0,0] # True Negative
    # fp_sum = MCM[:, 0, 1,1] # False Positive

    tp_sum = MCM[:, 1, 1]  # True Positive
    fn_sum = MCM[:, 1, 0]  # False Negative

    # 这里加1e-6，防止 0/0的情况计算得到nan，即tp_sum和fn_sum同时为0的情况
    Condition_negative = tp_sum[1] + fn_sum[1]

    Positive = tp_sum[1] / Condition_negative

    return Positive


def NegativeCalc(Predictions, Labels):
    MCM = multilabel_confusion_matrix(Labels, Predictions,
                                      sample_weight=None,
                                      labels=None, samplewise=None)
    # MCM此处是 5 * 2 * 2的混淆矩阵（ndarray格式），5表示的是5分类

    tp_sum = MCM[:, 1, 1]  # True Positive
    fn_sum = MCM[:, 1, 0]  # False Negative

    # 这里加1e-6，防止 0/0的情况计算得到nan，即tp_sum和fn_sum同时为0的情况
    # 这里加1e-6，防止 0/0的情况计算得到nan，即tp_sum和fn_sum同时为0的情况
    Condition_negative = tp_sum[0] + fn_sum[0]

    Negative = tp_sum[0] / Condition_negative

    return Negative


def sensitivity_auc_ci(y_true, y_score, positive=1):
    AUC = recall_score(y_true, y_score)
    N1 = sum(y_true == positive)
    N2 = sum(y_true != positive)
    Q1 = AUC / (2 - AUC)
    Q2 = 2 * AUC ** 2 / (1 + AUC)
    SE_AUC = sqrt((AUC * (1 - AUC) + (N1 - 1) * (Q1 - AUC ** 2) + (N2 - 1) * (Q2 - AUC ** 2)) / (N1 * N2))
    lower = AUC - 1.96 * SE_AUC
    upper = AUC + 1.96 * SE_AUC
    if lower < 0:
        lower = 0
    if upper > 1:
        upper = 1
    return '%.3f'%AUC + '(' + '%.3f'%lower + ',' + '%.3f'%upper + ')'


def specificity_auc_ci(y_true, y_score, positive=1):
    AUC = specificityCalc(y_true, y_score)
    N1 = sum(y_true == positive)
    N2 = sum(y_true != positive)
    Q1 = AUC / (2 - AUC)
    Q2 = 2 * AUC ** 2 / (1 + AUC)
    SE_AUC = sqrt((AUC * (1 - AUC) + (N1 - 1) * (Q1 - AUC ** 2) + (N2 - 1) * (Q2 - AUC ** 2)) / (N1 * N2))
    lower = AUC - 1.96 * SE_AUC
    upper = AUC + 1.96 * SE_AUC
    if lower < 0:
        lower = 0
    if upper > 1:
        upper = 1
    return '%.3f'%AUC + '(' + '%.3f'%lower + ',' + '%.3f'%upper + ')'


def Positive_ci(y_true, y_score, positive=1):
    AUC = PositiveCalc(y_true, y_score)
    N1 = sum(y_true == positive)
    N2 = sum(y_true != positive)
    Q1 = AUC / (2 - AUC)
    Q2 = 2 * AUC ** 2 / (1 + AUC)
    SE_AUC = sqrt((AUC * (1 - AUC) + (N1 - 1) * (Q1 - AUC ** 2) + (N2 - 1) * (Q2 - AUC ** 2)) / (N1 * N2))
    lower = AUC - 1.96 * SE_AUC
    upper = AUC + 1.96 * SE_AUC
    if lower < 0:
        lower = 0
    if upper > 1:
        upper = 1
    return '%.3f'%AUC + '(' + '%.3f'%lower + ',' + '%.3f'%upper + ')'


def Negative_ci(y_true, y_score, positive=1):
    AUC = NegativeCalc(y_true, y_score)
    N1 = sum(y_true == positive)
    N2 = sum(y_true != positive)
    Q1 = AUC / (2 - AUC)
    Q2 = 2 * AUC ** 2 / (1 + AUC)
    SE_AUC = sqrt((AUC * (1 - AUC) + (N1 - 1) * (Q1 - AUC ** 2) + (N2 - 1) * (Q2 - AUC ** 2)) / (N1 * N2))
    lower = AUC - 1.96 * SE_AUC
    upper = AUC + 1.96 * SE_AUC
    if lower < 0:
        lower = 0
    if upper > 1:
        upper = 1
    return '%.3f'%AUC + '(' + '%.3f'%lower + ',' + '%.3f'%upper + ')'


def Positive_likelihood_auc_ci(y_true, y_score, positive=1):
    AUC1 = recall_score(y_true, y_score)
    AUC2 = specificityCalc(y_true, y_score)
    print(AUC2)
    print(AUC1)
    AUC = AUC1 / (1 - AUC2)
    print(AUC)
    N1 = sum(y_true == positive)
    N2 = sum(y_true != positive)
    Q1 = AUC / (2 - AUC)
    Q2 = 2 * AUC ** 2 / (1 + AUC)
    print((AUC * (1 - AUC) + (N1 - 1) * (Q1 - AUC ** 2) + (N2 - 1) * (Q2 - AUC ** 2)) / (N1 * N2))
    SE_AUC = sqrt(abs((AUC * (1 - AUC) + (N1 - 1) * (Q1 - AUC ** 2) + (N2 - 1) * (Q2 - AUC ** 2)) / (N1 * N2)))
    lower = AUC - 1.96 * SE_AUC
    upper = AUC + 1.96 * SE_AUC
    # if lower < 0:
    #     lower = 0
    # if upper > 1:
    #     upper = 1
    return '%.3f'%AUC + '(' + '%.3f'%lower + ',' + '%.3f'%upper + ')'


# 阳性似然比=敏感性/ (1-特异性)
# 阴性似然比 = （1-灵敏度）/特异度
# 阳性似然比 = 灵敏度 / (1 − 特异性) ≈ 0.67 / (1 − 0.91) ≈ 7.4
# 阴性似然比 = (1 − 敏感性) / 特异性 ≈ (1 − 0.67) / 0.91 ≈ 0.37


def Negative_likelihood_auc_ci(y_true, y_score, positive=1):
    AUC1 = recall_score(y_true, y_score)
    AUC2 = specificityCalc(y_true, y_score)
    AUC2 = 0.1 if AUC2 == 0  else AUC2
    AUC = (1 - AUC1) / AUC2
    N1 = sum(y_true == positive)
    N2 = sum(y_true != positive)
    Q1 = AUC / (2 - AUC)
    Q2 = 2 * AUC ** 2 / (1 + AUC)
    SE_AUC = sqrt((AUC * (1 - AUC) + (N1 - 1) * (Q1 - AUC ** 2) + (N2 - 1) * (Q2 - AUC ** 2)) / (N1 * N2))
    lower = AUC - 1.96 * SE_AUC
    upper = AUC + 1.96 * SE_AUC
    if lower < 0:
        lower = 0
    if upper > 1:
        upper = 1
    return '%.3f'%AUC + '(' + '%.3f'%lower + ',' + '%.3f'%upper + ')'


def accuracy_ci(y_true, y_score, positive=1):
    AUC = accuracy_score(y_true, y_score)
    N1 = sum(y_true == positive)
    N2 = sum(y_true != positive)
    Q1 = AUC / (2 - AUC)
    Q2 = 2 * AUC ** 2 / (1 + AUC)
    SE_AUC = sqrt((AUC * (1 - AUC) + (N1 - 1) * (Q1 - AUC ** 2) + (N2 - 1) * (Q2 - AUC ** 2)) / (N1 * N2))
    lower = AUC - 1.96 * SE_AUC
    upper = AUC + 1.96 * SE_AUC
    if lower < 0:
        lower = 0
    if upper > 1:
        upper = 1
    return '%.3f'%AUC + '(' + '%.3f'%lower + ',' + '%.3f'%upper + ')'


def f1_ci(y_true, y_score, positive=1):
    AUC = f1_score(y_true, y_score)
    N1 = sum(y_true == positive)
    N2 = sum(y_true != positive)
    Q1 = AUC / (2 - AUC)
    Q2 = 2 * AUC ** 2 / (1 + AUC)
    SE_AUC = sqrt((AUC * (1 - AUC) + (N1 - 1) * (Q1 - AUC ** 2) + (N2 - 1) * (Q2 - AUC ** 2)) / (N1 * N2))
    lower = AUC - 1.96 * SE_AUC
    upper = AUC + 1.96 * SE_AUC
    if lower < 0:
        lower = 0
    if upper > 1:
        upper = 1
    return '%.3f'%AUC + '(' + '%.3f'%lower + ',' + '%.3f'%upper + ')'




df_result = pd.DataFrame(
    columns=['AUC', 'Sensitivity', 'Specificity', 'Positive_predictive_value', 'Negative_predictive_value',
            'Positive_likelihood_ratio', 'Negative_likelihood_ratio', 'accuracy', 'F1_Score'])


df = pd.read_csv('train_proba_all.csv')
df = df.drop(['Unnamed: 0'],axis = 1)
columns = df.columns

for i in  columns:
    if i == 'y_true':
        continue
    i_name = i + '1'
    if i == 'y_xgb_train':
        df[i_name] = df[i].apply(lambda x: 1 if x > 0.502811 else 0)
    else:
        df[i_name] = df[i].apply(lambda x: 1 if x > 0.5 else 0)
    result_li = []
    result_li.append(roc_auc_ci(df['y_true'], df[i], positive=1))

    result_li.append(sensitivity_auc_ci(df['y_true'], df[i_name], positive=1))

    result_li.append(specificity_auc_ci(df['y_true'], df[i_name], positive=1))

    result_li.append(Positive_ci(df['y_true'], df[i_name], positive=1))

    result_li.append(Negative_ci(df['y_true'], df[i_name], positive=1))

    result_li.append(Positive_likelihood_auc_ci(df['y_true'], df[i_name], positive=1))

    result_li.append(Negative_likelihood_auc_ci(df['y_true'], df[i_name], positive=1))

    result_li.append(accuracy_ci(df['y_true'], df[i_name], positive=1))

    result_li.append(f1_ci(df['y_true'], df[i_name], positive=1))
    print(result_li)
    df_result.loc[len(df_result.index)] = result_li
df_result.loc[len(df_result.index)] = [0] * 9

df = pd.read_csv('val_proba_all.csv')
df = df.drop(['Unnamed: 0'],axis = 1)
columns = df.columns

print(columns.tolist())

for i in columns:
    if i == 'y_true':
        continue
    i_name = i + '1'
    if i == 'y_xgb_train':
        df[i_name] = df[i].apply(lambda x: 1 if x > 0.496 else 0)
    else:
        df[i_name] = df[i].apply(lambda x: 1 if x > 0.5 else 0)
    result_li = []
    result_li.append(roc_auc_ci(df['y_true'], df[i], positive=1))

    result_li.append(sensitivity_auc_ci(df['y_true'], df[i_name], positive=1))

    result_li.append(specificity_auc_ci(df['y_true'], df[i_name], positive=1))

    result_li.append(Positive_ci(df['y_true'], df[i_name], positive=1))

    result_li.append(Negative_ci(df['y_true'], df[i_name], positive=1))

    result_li.append(Positive_likelihood_auc_ci(df['y_true'], df[i_name], positive=1))

    result_li.append(Negative_likelihood_auc_ci(df['y_true'], df[i_name], positive=1))

    result_li.append(accuracy_ci(df['y_true'], df[i_name], positive=1))

    result_li.append(f1_ci(df['y_true'], df[i_name], positive=1))
    df_result.loc[len(df_result.index)] = result_li
df_result.loc[len(df_result.index)] = [0] * 9

df = pd.read_csv('test_proba_all.csv')
df = df.drop(['Unnamed: 0'],axis = 1)
columns = df.columns

for i in columns:
    print(i)
    if i == 'y_true':
        continue
    i_name = i + '1'
    if i == 'y_xgb_train':
        df[i_name] = df[i].apply(lambda x: 1 if x > 0.502811 else 0)
    else:
        df[i_name] = df[i].apply(lambda x: 1 if x > 0.5 else 0)
    result_li = []
    result_li.append(roc_auc_ci(df['y_true'], df[i], positive=1))

    result_li.append(sensitivity_auc_ci(df['y_true'], df[i_name], positive=1))
    print(i_name)

    result_li.append(specificity_auc_ci(df['y_true'], df[i_name], positive=1))

    result_li.append(Positive_ci(df['y_true'], df[i_name], positive=1))

    result_li.append(Negative_ci(df['y_true'], df[i_name], positive=1))

    result_li.append(Positive_likelihood_auc_ci(df['y_true'], df[i_name], positive=1))

    result_li.append(Negative_likelihood_auc_ci(df['y_true'], df[i_name], positive=1))

    result_li.append(accuracy_ci(df['y_true'], df[i_name], positive=1))

    result_li.append(f1_ci(df['y_true'], df[i_name], positive=1))
    df_result.loc[len(df_result.index)] = result_li
df_result.loc[len(df_result.index)] = [0] * 9


df = pd.read_csv('shhs_proba_all.csv')
df = df.drop(['Unnamed: 0'],axis = 1)
columns = df.columns

for i in columns:
    print(i)
    if i == 'y_true':
        continue
    i_name = i + '1'
    if i == 'y_xgb_train':
        df[i_name] = df[i].apply(lambda x: 1 if x > 0.502811 else 0)
    else:
        df[i_name] = df[i].apply(lambda x: 1 if x > 0.5 else 0)
    result_li = []
    result_li.append(roc_auc_ci(df['y_true'], df[i], positive=1))

    result_li.append(sensitivity_auc_ci(df['y_true'], df[i_name], positive=1))
    print(i_name)

    result_li.append(specificity_auc_ci(df['y_true'], df[i_name], positive=1))

    result_li.append(Positive_ci(df['y_true'], df[i_name], positive=1))

    result_li.append(Negative_ci(df['y_true'], df[i_name], positive=1))

    result_li.append(Positive_likelihood_auc_ci(df['y_true'], df[i_name], positive=1))

    result_li.append(Negative_likelihood_auc_ci(df['y_true'], df[i_name], positive=1))

    result_li.append(accuracy_ci(df['y_true'], df[i_name], positive=1))

    result_li.append(f1_ci(df['y_true'], df[i_name], positive=1))
    df_result.loc[len(df_result.index)] = result_li
df_result.loc[len(df_result.index)] = [0] * 9

df_result.to_csv('zhibiao.csv',index = False)


# AUC
# Sensitivity 灵敏度
# Specificity 特异度
# Positive predictive value 阳性预测数
# Negative predictivevalue 阴性预测值
# Positive likelihood ratio
# Negative likelihood ratio

# 阳性似然比=敏感性/ (1-特异性)
# 阴性似然比 = （1-灵敏度）/特异度

# 符合率
# F1-Score
