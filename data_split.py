import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, train_test_split

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, \
    classification_report, roc_curve, auc

import pyreadr
from sklearn.preprocessing import StandardScaler


L = ['BMI', 'tunwei', 'wc', 'LAP', 'VAI', 'CVAI', 'ABSI', 'BRI',
     'BAE', 'PI', 'RFM', 'CI', 'AVI', 'WHR', 'WHtR', 'Lean_body_mass', 'Fat_mass', 'Percent_fat', 'NC',
'gender','age_diagnose',]
print(len(L))

feature_columns = [
    'CT', 'xueyang_low',
    'MSPO_total',
                   'BMI', 'tunwei', 'wc', 'LAP', 'VAI', 'CVAI', 'ABSI', 'BRI',
     'BAE', 'PI', 'RFM', 'CI', 'AVI', 'WHR', 'WHtR', 'Lean_body_mass', 'Fat_mass', 'Percent_fat', 'NC',
'c_LDL', 'c_TG', 'c_HDL', 'zongdgc', 'c_TC',
     'c_xuezhi_2','c_FPG',  'c_SBP',  'c_DBP','disease_gaoxueya', 'disease_gaoxuezhi', 'disease_dm', 'drug_jiangtang', 'drug_jiangya',
     'drug_jiangzhi', 'smoke', 'alcohol', 'age_diagnose','c_gaoyueya_2',
'gender','zzdb_A','zzdb_B','zzdb_E','zhidanbai_a'

# 'yidao','LDL','TG', 'HDL',
# 'c_LDL', 'c_TG', 'c_HDL', 'zongdgc', 'c_TC',
#      'c_xuezhi_1', 'c_xuezhi_2', 'xuetang', 'c_FPG', 'SBP', 'c_SBP', 'DBP', 'c_DBP', 'c_gaoyueya_1',
#      'c_gaoyueya_2', 'disease_gaoxueya', 'disease_gaoxuezhi', 'disease_dm', 'drug_jiangtang', 'drug_jiangya',
#      'drug_jiangzhi', 'smoke', 'alcohol', 'age_diagnose',
                   ]

DM = pyreadr.read_r('osa_include_BMI.rds')  # also works for RData
train = DM[None]
print(len(train))
print(train['S_AHI_2'].value_counts())

print(train['ruyuan_date'].value_counts())

# train['ruyuan_year'] = train['ruyuan_date'].apply(lambda x: int(str(x)[:4]))
#
# train['group'] = train['ruyuan_year'].apply(lambda x:2 if x>=2020 else 1)

train['ruyuan_year'] = train['ruyuan_date'].apply(lambda x: int(str(x)[:4]))

# train['group'] = train['ruyuan_year'].apply(lambda x:2 if x>=2020 else 1)
#
# train.to_csv('osa_group_fill.csv',index = False)

# train['ruyuan_year_mouth'] = train['ruyuan_date'].apply(lambda x: int(str(x)[:6]))
#


test = train[train['ruyuan_year'] >= 2020]

train = train[train['ruyuan_year'] < 2020]

train,label = train[feature_columns ],train['S_AHI_2']

train_data, val_data, train_label, val_label = train_test_split(train, label, random_state=2, test_size=0.2,
                                                                    stratify=label)