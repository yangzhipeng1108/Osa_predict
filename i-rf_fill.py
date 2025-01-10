import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, train_test_split
import gc
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, auc
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import os
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, \
    classification_report, roc_curve, auc
from itertools import cycle
from sklearn.preprocessing import OneHotEncoder
import pyreadr
from sklearn.preprocessing import PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor

feature_columns = [    'BMI', 'tunwei', 'wc',  'ABSI', 'BRI',
     'BAE', 'PI', 'RFM', 'CI', 'AVI', 'WHR', 'WHtR', 'Lean_body_mass', 'Fat_mass', 'Percent_fat', 'NC',
'gender','age_diagnose',
    'LAP','VAI','CVAI','TG', 'HDL'
                       ]
# 'meat'  ,  osa_ML.rds  SH_obesity.rds
DM = pyreadr.read_r('osa_ML.rds')  # also works for RData
train = DM[None]
print(len(train))
# train['ruyuan_year'] = train['ruyuan_date'].apply(lambda x: int(str(x)[:4]))
# train = train[train['ruyuan_year'] >= 2020]

train = train[feature_columns + ['S_AHI_2']]

for i in train.columns:
    train[i] = train[i].astype('float')

feature_fill_columns = []
for i in feature_columns:
    if train[i].isnull().sum() > 0:
        feature_fill_columns.append(i)
print(feature_fill_columns)

train = train.reset_index(drop=True)

X_full, y_full = train[feature_fill_columns], train['S_AHI_2']
n_samples = X_full.shape[0]  # 样本
n_features = X_full.shape[1]  # 特征
print(n_samples)
print(n_features)

# 首先确定我们希望放入的缺失值数据的比例，在这里我假设是50%，可以自己改动
rng = np.random.RandomState(0)
missing_rate = 0.5
n_missing_samples = int(np.floor(n_samples * n_features * missing_rate))
# np.floor  向下取整
# 所有数据要随机遍布在数据集的各行各列当中，而一个确实的数据会需要一盒行索引和一个列索引
# 如果能够创造一个数组，就可以利用索引来赋空值

X_missing_reg = X_full.copy()
# 查看缺失情况
missing = X_missing_reg.isna().sum()
missing = pd.DataFrame(data={'特征': missing.index, '缺失值个数': missing.values})
# 通过~取反，选取不包含数字0的行
missing = missing[~missing['缺失值个数'].isin([0])]
# 缺失比例
missing['缺失比例'] = missing['缺失值个数'] / X_missing_reg.shape[0]
X_df = X_missing_reg.isnull().sum()
# 得出列名 缺失值最少的列名 到 缺失值最多的列名
colname = X_df[~X_df.isin([0])].sort_values().index.values
# 缺失值从小到大的特征顺序
sortindex = []

for i in colname:
    sortindex.append(X_missing_reg.columns.tolist().index(str(i)))
# 遍历所有的特征，从缺失最少的开始进行填补，每完成一次回归预测，就将预测值放到原本的特征矩阵中，再继续填补下一个特征
for i in sortindex:
    # 构建我们的新特征矩阵和新标签
    df = X_missing_reg  # 充当中间数据集
    fillc = df.iloc[:, i]  # 缺失值最少的特征列
    # 除了第 i 特征列，剩下的特征列+原有的完整标签 = 新的特征矩阵
    df = pd.concat([df.drop(df.columns[i], axis=1), pd.DataFrame(y_full)], axis=1)
    # 在新特征矩阵中，对含有缺失值的列，进行0的填补 ，没循环一次，用0填充的列越来越少
    df_0 = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0).fit_transform(df)
    # 找出训练集和测试集
    # 标签
    Ytrain = fillc[fillc.notnull()]  # 没有缺失的部分，就是 Y_train
    Ytest = fillc[fillc.isnull()]  # 不是需要Ytest的值，而是Ytest的索引
    # 特征矩阵
    Xtrain = df_0[Ytrain.index, :]
    Xtest = df_0[Ytest.index, :]  # 有缺失值的特征情况
    rfc = RandomForestRegressor(n_estimators=100)  # 实例化
    rfc = rfc.fit(Xtrain, Ytrain)  # 训练
    Ypredict = rfc.predict(Xtest)  # 预测结果，就是要填补缺失值的值
    # 将填补好的特征返回到我们的原始的特征矩阵中
    X_missing_reg.loc[X_missing_reg.iloc[:, i].isnull(), X_missing_reg.columns[i]] = Ypredict

train[feature_fill_columns] = X_missing_reg

train.to_csv('osa_zxy.csv', index=False)

# SH_obesity.csv  osa_zxy.csv