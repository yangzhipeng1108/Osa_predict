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
    classification_report, roc_curve, auc,brier_score_loss
from itertools import cycle
from sklearn.preprocessing import OneHotEncoder
import pyreadr



feature_columns = [

                   'BMI', 'tunwei', 'wc',  'ABSI', 'BRI',
     'BAE', 'PI', 'RFM', 'CI', 'AVI', 'WHR', 'WHtR', 'Lean_body_mass', 'Fat_mass', 'Percent_fat', 'NC',
'gender','age_diagnose',
    'LAP','VAI','CVAI',
# 'c_LDL', 'c_TG', 'c_HDL', 'zongdgc', 'c_TC',
#      'c_xuezhi_2','c_FPG',  'c_SBP',  'c_DBP','disease_gaoxueya', 'disease_gaoxuezhi', 'disease_dm', 'drug_jiangtang', 'drug_jiangya',
#      'drug_jiangzhi', 'smoke', 'alcohol'
#     ,'c_gaoyueya_2',
# 'zzdb_A','zzdb_B','zzdb_E','zhidanbai_a'
                   ]

# from sklearn.preprocessing import PolynomialFeatures
# ploy = PolynomialFeatures(degree = 2)
DM = pyreadr.read_r('osa_ML.rds')  # also works for RData
train = DM[None]
print(len(train))

# osa_train = pd.read_csv('osa_zxy_train.csv')
# osa_test= pd.read_csv('osa_zxy_test.csv')
# train['ruyuan_year'] = train['ruyuan_date'].apply(lambda x: int(str(x)[:4]))
# test = train[train['ruyuan_year'] >= 2020]
#
#
# train = train[train['ruyuan_year'] < 2020]
# # train = train[train['ruyuan_year'] >= 2010]
# print(train['S_AHI_2'].value_counts())
#
# train[feature_columns] = osa_train[feature_columns]
# test[feature_columns] = osa_test[feature_columns]
# train = pd.concat([train,test],axis =0)

osa = pd.read_csv('osa_zxy.csv')
train[feature_columns] = osa[feature_columns]  # also works for RData


# st = StandardScaler()
# train[feature_columns] = pd.DataFrame(st.fit_transform(train[feature_columns]))

# for i in feature_columns:
#     train[i] = train[i].astype('float')
#
# train_BIM = train[['BMI', 'tunwei', 'wc', 'LAP', 'VAI', 'CVAI', 'ABSI', 'BRI',
#      'BAE', 'PI', 'RFM', 'CI', 'AVI', 'WHR', 'WHtR', 'Lean_body_mass', 'Fat_mass', 'Percent_fat', 'NC']]
#
# print(train_BIM.isna().sum())
#
# train_BIM = pd.DataFrame(ploy.fit_transform(train_BIM))

# 'gender','age_diagnose',
# 'c_LDL', 'c_TG', 'c_HDL', 'zongdgc', 'c_TC',
#      'c_xuezhi_2','c_FPG',  'c_SBP',  'c_DBP','disease_gaoxueya', 'disease_gaoxuezhi', 'disease_dm', 'drug_jiangtang', 'drug_jiangya',
#      'drug_jiangzhi', 'smoke', 'alcohol'
#     ,'c_gaoyueya_2',
# 'zzdb_A','zzdb_B','zzdb_E','zhidanbai_a',

# train = train[['Patid','S_AHI_2','ruyuan_date']]
# train = pd.concat([train,train_BIM],axis = 1)


train['ruyuan_year'] = train['ruyuan_date'].apply(lambda x: int(str(x)[:4]))

# train['group'] = train['ruyuan_year'].apply(lambda x:2 if x>=2020 else 1)
#
# train.to_csv('osa_group_fill.csv',index = False)

# train['ruyuan_year_mouth'] = train['ruyuan_date'].apply(lambda x: int(str(x)[:6]))
#



test = train[train['ruyuan_year'] >= 2019]




train = train[train['ruyuan_year'] < 2019]
train = train[train['ruyuan_year'] >= 2007]
print(train['S_AHI_2'].value_counts())

train,label = train[feature_columns ],train['S_AHI_2']

train_data, val_data, train_label, val_label = train_test_split(train, label, random_state=2, test_size=0.2,
                                                                    stratify=label)

# train_data =pd.DataFrame(train_data,columns=feature_columns)
# train_label = pd.DataFrame(train_label,columns=['S_AHI_2'])
#
# val_data =pd.DataFrame(val_data,columns=feature_columns)
# val_label = pd.DataFrame(val_label,columns=['S_AHI_2'])
#
# train_result = pd.concat([train_data,train_label],axis = 1)
# train_result['train'] = 1
# val_result = pd.concat([val_data,val_label],axis = 1)
# val_result['train'] = 2
#
# train_result = pd.concat([train_result,val_result],axis = 0)
# train_result.to_csv('train_result.csv',index = False)

test_1 = test[test['S_AHI_2'] == 1]
test_1 = test_1.sample(n=201,random_state=2)
test_2 = test[test['S_AHI_2'] == 0]
test_2 = test_2.sample(n=100,random_state=2)

print(len(test_1))
print(len(test_2))
# 476
# 169
test = pd.concat([test_1,test_2],axis = 0)
print(test.columns)

test_data, test_label =  test[feature_columns],test['S_AHI_2']
# print(train_data.columns.tolist())

shhs  = pd.read_csv('SH_obesity.csv')


# shhs_1 = shhs[shhs['S_AHI_2'] == 1]
# shhs_1 = shhs_1.sample(n=2000,random_state=2020)
# shhs_2 = shhs[shhs['S_AHI_2'] == 0]
# shhs_2 = shhs_2.sample(n=1000,random_state=2020)
# shhs = pd.concat([shhs_1,shhs_2],axis = 0)
shhs_data, shhs_label =  shhs[feature_columns],shhs['S_AHI_2']


def eval_fn(data, predict):
    ###准确度
    print("---准确度---")
    print(round(accuracy_score(data, predict), 5))
    ###精确度
    print("---精确度---")
    print(round(precision_score(data, predict), 5))
    ###召回率
    print("---召回率---")
    print(round(recall_score(data, predict), 5))
    ###f1
    print("---f1---")
    print(round(f1_score(data, predict), 5))
    ###混淆矩阵
    print(confusion_matrix(data, predict))
    ###分类报告
    print(classification_report(data, predict))

    print("---AUC---")
    fpr, tpr, thresholds = roc_curve(data, predict)
    print(auc(fpr, tpr))


def train_lgb(train, label):
    '''
    模型训练
    '''
    # 划分模型
    train_x_tr, train_x_val, train_y_tr, train_y_val = train_test_split(train, label, random_state=0, test_size=0.2,
                                                                        stratify=label)

    # 设定类别变量
    # categorical_feature = [ ]
    # 建立特征重要性DataFrame
    imp = pd.DataFrame()
    imp['feat'] = train_x_tr.columns.tolist()
    # 预测结果初步构建
    oof_train = np.zeros((len(train_x_tr), 2))

    # 需要将dataframe格式的数据转化为矩阵形式
    xgtrain = xgb.DMatrix(train_x_tr.values, train_y_tr.values)
    xgtest = xgb.DMatrix(train_x_val.values, train_y_val.values)

    # ####### 二分类
    # params = {'max_depth': 5, 'eta': 0.1, 'silent': 1, 'subsample': 0.7, 'colsample_bytree': 0.7,
    #          'objective': 'binary:logistic'}
    #
    # # 设定watchlist用于查看模型状态
    # watchlist = [(xgtest, 'eval'), (xgtrain, 'train')]
    # num_round = 10
    # bst = xgb.train(params, xgtrain, num_round, watchlist)
    #
    # # 使用模型预测
    # preds = bst.predict(xgtest)
    #
    # vals_y = [1 if i >= 0.55 else 0 for i in preds]
    #
    #
    # eval_fn(train_y_val,vals_y)
    #
    # print("---test---")
    # test_data_matrix = xgb.DMatrix(test_data.values)
    # test_predict = bst.predict(test_data_matrix)
    # test_predict = [1 if i >= 0.58 else 0 for i in test_predict]
    #
    # eval_fn(test_label, test_predict)



    ####### 多分类
    # 参数设定
    params = {
        'booster': 'gbtree',
        'objective': 'multi:softmax',  # 多分类的问题
        'num_class': 2,  # 类别数，与 multisoftmax 并用
        'gamma': 0.1,  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
        'max_depth': 12,  # 构建树的深度，越大越容易过拟合
        'lambda': 2,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
        'subsample': 0.7,  # 随机采样训练样本
        'colsample_bytree': 0.7,  # 生成树时进行的列采样
        'min_child_weight': 3,
        'silent': 1,  # 设置成1则没有运行信息输出，最好是设置为0.
        'eta': 0.007,  # 如同学习率
        'seed': 1000,
        'nthread': 4,  # cpu 线程数
    }

    # 设定watchlist用于查看模型状态
    watchlist = [(xgtest, 'eval'), (xgtrain, 'train')]
    num_round = 10
    # bst = xgb.train(params, xgtrain, num_round, watchlist)

    model = xgb.XGBClassifier(**params)

    model.fit(train_x_tr, train_y_tr, eval_set=[(train_x_val, train_y_val)], eval_metric="auc", early_stopping_rounds=10,
              verbose=True)

    import shap
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(test_data)
    shap_values2 = explainer(test_data)
    print(shap_values2)
    shap.plots.bar(shap_values2)
    # fig.get_figure().savefig('shap.png')  # 保存图片

    # print(bst.get_score(importance_type='gain'))

    feature_importance = pd.DataFrame()
    feature_importance['name'] = train.columns.tolist()
    feature_importance['value'] = model.feature_importances_
    feature_importance.to_csv('feature_importance.csv',index=False)

    # plot_importance(model,max_num_features=10)
    # pyplot.show()

    # plot feature importance

    # plot_importance(model)
    # pyplot.show()
    print("---train---")
    # 使用模型预测
    preds = model.predict_proba(train_x_val)
    # preds = bst.predict(xgtest)

    print("---train---")
    new_df = pd.DataFrame()
    new_df.index = train_y_val.index
    new_df['y_xgb_predict'] = preds[:, 1]
    new_df['y_true'] = train_y_val

    new_df.to_csv('train_proba_xgb.csv')
    print("---AUC-proba---")
    fpr, tpr, thresholds = roc_curve(train_y_val, preds[:, 1])
    roc_auc = auc(fpr, tpr)
    print(roc_auc)


    preds = np.argmax(preds, axis=1)

    eval_fn(train_y_val, preds)


    print("---val---")
    val_predict = model.predict_proba(val_data)

    new_df = pd.DataFrame()
    new_df.index = val_label.index
    new_df['y_xgb_predict'] = val_predict[:, 1]
    new_df['y_true'] = val_label

    new_df.to_csv('val_proba_xgb.csv')

    print("---AUC-proba---")
    fpr, tpr, thresholds = roc_curve(val_label, val_predict[:, 1])
    roc_auc = auc(fpr, tpr)
    print(roc_auc)

    val_predict = np.argmax(val_predict, axis=1)


    eval_fn(val_label, val_predict)


    print("---test---")
    test_predict = model.predict_proba(test_data)

    new_df = pd.DataFrame()
    new_df.index = test_label.index
    new_df['y_xgb_predict'] = test_predict[:, 1]
    new_df['y_true'] = test_label

    new_df.to_csv('test_proba_xgb.csv')

    print("---AUC-proba---")
    fpr, tpr, thresholds = roc_curve(test_label, test_predict[:, 1])
    roc_auc = auc(fpr, tpr)
    print(roc_auc)

    test_predict = np.argmax(test_predict, axis=1)


    eval_fn(test_label, test_predict)

    print("---shhs---")
    shhs_predict = model.predict_proba(shhs_data)

    new_df = pd.DataFrame()
    new_df.index = shhs_label.index
    new_df['y_xgb_predict'] = shhs_predict[:, 1]
    new_df['y_true'] = shhs_label

    new_df.to_csv('shhs_proba_xgb.csv')

    print("---AUC-proba---")
    fpr, tpr, thresholds = roc_curve(shhs_label, shhs_predict[:, 1])
    roc_auc = auc(fpr, tpr)
    print(roc_auc)

    shhs_predict = np.argmax(shhs_predict, axis=1)


    eval_fn(shhs_label, shhs_predict)


train_lgb(train_data, train_label)
