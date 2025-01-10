import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
import time
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, \
    classification_report, roc_curve, auc
import torch
import torch.nn as nn
import torch.nn.functional
import torch.optim as optim
import torch.utils.data as Data
from sklearn.preprocessing import StandardScaler
import pyreadr

torch.manual_seed(1)  # reproducible


#     'NC','tunwei','height','wc','weight','TG','HDL'
# 'gender','age_diagnose',

feature_columns = [

                   'BMI', 'tunwei', 'wc', 'LAP', 'VAI', 'CVAI', 'ABSI', 'BRI',
     'BAE', 'PI', 'RFM', 'CI', 'AVI', 'WHR', 'WHtR', 'Lean_body_mass', 'Fat_mass', 'Percent_fat', 'NC',
'gender','age_diagnose',
    # 'c_gaoyueya_2', 'smoke', 'alcohol','c_FPG',  'c_SBP',  'c_DBP',
# 'c_LDL', 'c_TG', 'c_HDL', 'zongdgc', 'c_TC',
# 'c_LDL', 'c_TG', 'c_HDL', 'zongdgc', 'c_TC',
#      'c_xuezhi_2','c_FPG',  'c_SBP',  'c_DBP','disease_gaoxueya', 'disease_gaoxuezhi', 'disease_dm', 'drug_jiangtang', 'drug_jiangya',
#      'drug_jiangzhi', 'smoke', 'alcohol'
#     ,'c_gaoyueya_2',
# 'zzdb_A','zzdb_B','zzdb_E','zhidanbai_a','Patid'
                   ]

# from sklearn.preprocessing import PolynomialFeatures
# ploy = PolynomialFeatures(degree = 2)
DM = pyreadr.read_r('osa_for_ML.rds')  # also works for RData
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
st = StandardScaler()
train[feature_columns] = pd.DataFrame(st.fit_transform(train[feature_columns]))

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
test_1 = test_1.sample(n=201,random_state=629)
test_2 = test[test['S_AHI_2'] == 0]
# test_2 = test_2.sample(n=100,random_state=2)

print(len(test_1))
print(len(test_2))
# 476
# 169
test = pd.concat([test_1,test_2],axis = 0)

test_data, test_label =  test[feature_columns],test['S_AHI_2']
# print(train_data.columns.tolist())



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


class AutoEncoder_classifier(nn.Module):
    def __init__(self, in_dim=8, out_dim=8):
        super(AutoEncoder_classifier, self).__init__()
        self.encode = nn.Sequential(
            nn.Linear(in_features=in_dim, out_features=200),
            nn.Dropout(p=0.1),
            nn.Linear(in_features=200, out_features=800),
            nn.Dropout(p=0.1),
            nn.Linear(in_features=800, out_features=3200),
            nn.Dropout(p=0.1),
            # nn.Linear(in_features=2400, out_features=6400),
            # nn.Dropout(p=0.1),
        )

        self.decoder = nn.Sequential(
            # nn.Linear(in_features=6400, out_features=2400),
            # nn.Dropout(p=0.1),
            nn.Linear(in_features=3200, out_features=400),
            nn.Dropout(p=0.1),
            nn.Linear(in_features=400, out_features=in_dim),
            nn.Dropout(p=0.1),
            nn.Linear(in_features=in_dim, out_features=out_dim),
        )

    def forward(self, x):
        encoder_out = self.encode(x)

        decoder_out = self.decoder(encoder_out)
        # import torch.nn.functional as F
        # decoder_out = F.softmax(decoder_out, dim=1)
        return decoder_out


batch_size = 100
num_epochs = 50
in_dim = 21
out_dim = 2

train_x_tr, train_x_val, train_y_tr, train_y_val = train_test_split(train, label, random_state=2, test_size=0.2,
                                                                    stratify=label)

autoEncoder = AutoEncoder_classifier(in_dim=in_dim, out_dim=out_dim)
if torch.cuda.is_available():
    autoEncoder.cuda()  # 注:将模型放到GPU上,因此后续传入的数据必须也在GPU上

Loss = nn.CrossEntropyLoss()
Optimizer = optim.Adam(autoEncoder.parameters(), lr=0.001)

# 定义期望平均激活值和KL散度的权重

torch_dataset = Data.TensorDataset(torch.from_numpy(train_x_tr.values).to(torch.float32),
                                   torch.from_numpy(train_y_tr.values).to(torch.float32))

# 把 dataset 放入 DataLoader
data_loader = Data.DataLoader(
    dataset=torch_dataset,  # torch TensorDataset format
    batch_size=batch_size,  # mini batch size
    shuffle=True,  # 要不要打乱数据 (打乱比较好)
    # num_workers=2,  # 多线程来读数据
)

best_auc = 0
for epoch in range(num_epochs):
    t_epoch_start = time.time()
    for i, (image_batch, image_label) in enumerate(data_loader):
        # flatten batch
        image_batch = image_batch.view(image_batch.size(0), -1)
        if torch.cuda.is_available():
            image_batch = image_batch.cuda()
            image_label = image_label.cuda()
        predict = autoEncoder(image_batch)

        Optimizer.zero_grad()
        loss = Loss(predict, image_label.long())
        loss.backward()
        Optimizer.step()

        if (i + 1) % 20 == 0:
            print('Epoch {}/{}, Iter {}/{}, loss: {:.4f}, time: {:.2f}s'.format(
                epoch + 1, num_epochs, (i + 1), len(train_data) // batch_size, loss.data, time.time() - t_epoch_start
            ))

    preds = autoEncoder(torch.from_numpy(train_x_val.values).to(torch.float32).cuda()).cpu()
    import torch.nn.functional as F

    preds = F.softmax(preds, dim=1)
    preds = preds.detach().numpy()

    fpr, tpr, thresholds = roc_curve(train_y_val.values, preds[:, 1])
    auc_value = auc(fpr, tpr)

    if auc_value > best_auc:
        best_auc = auc_value

        torch.save(autoEncoder, 'AutoEncoder.pkl')
        print('________________________________________')
        print('finish training')
        print(auc_value)


Coder = AutoEncoder_classifier(in_dim=in_dim, out_dim=1)
Coder = torch.load('AutoEncoder.pkl')

print('train')
train_predict = Coder(torch.from_numpy(train_x_val.values).to(torch.float32).cuda()).cpu()
import torch.nn.functional as F

train_predict = F.softmax(train_predict, dim=1)
train_predict = train_predict.detach().numpy()
new_df = pd.DataFrame()
new_df.index = train_y_val.index
new_df['y_cnn_predict'] = train_predict[:, 1]
new_df['y_true'] = train_y_val

new_df.to_csv('predict_proba_cnn.csv')
print("---AUC-proba---")
fpr, tpr, thresholds = roc_curve(train_y_val, train_predict[:, 1])
roc_auc = auc(fpr, tpr)
print(roc_auc)
train_predict = np.argmax(train_predict, axis=1)

new_df = pd.DataFrame()
new_df.index = train_y_val.index
new_df['y_cnn_predict'] = train_predict
new_df['y_true'] = train_y_val


eval_fn(train_y_val, train_predict)


print('val')
val_predict = Coder(torch.from_numpy(val_data.values).to(torch.float32).cuda()).cpu()
import torch.nn.functional as F

val_predict = F.softmax(val_predict, dim=1)
val_predict = val_predict.detach().numpy()
new_df = pd.DataFrame()
new_df.index = val_label.index
new_df['y_cnn_predict'] = val_predict[:, 1]
new_df['y_true'] = val_label

new_df.to_csv('predict_proba_cnn.csv')
print("---AUC-proba---")
fpr, tpr, thresholds = roc_curve(val_label, val_predict[:, 1])
roc_auc = auc(fpr, tpr)
print(roc_auc)
val_predict = np.argmax(val_predict, axis=1)

new_df = pd.DataFrame()
new_df.index = val_label.index
new_df['y_cnn_predict'] = val_predict
new_df['y_true'] = val_label


eval_fn(val_label, val_predict)




print('test')
test_predict = Coder(torch.from_numpy(test_data.values).to(torch.float32).cuda()).cpu()
import torch.nn.functional as F

test_predict = F.softmax(test_predict, dim=1)
test_predict = test_predict.detach().numpy()
new_df = pd.DataFrame()
new_df.index = test_label.index
new_df['y_cnn_predict'] = test_predict[:, 1]
new_df['y_true'] = test_label

new_df.to_csv('predict_proba_cnn.csv')
print("---AUC-proba---")
fpr, tpr, thresholds = roc_curve(test_label, test_predict[:, 1])
roc_auc = auc(fpr, tpr)
print(roc_auc)
test_predict = np.argmax(test_predict, axis=1)

new_df = pd.DataFrame()
new_df.index = test_label.index
new_df['y_cnn_predict'] = test_predict
new_df['y_true'] = test_label


eval_fn(test_label, test_predict)


# def kl_1(p, q):
#     p = torch.nn.functional.softmax(p, dim=-1)
#     _kl = torch.sum(p*(torch.log_softmax(p,dim=-1)) - torch.nn.functional.log_softmax(q, dim=-1),1)
#     return torch.mean(_kl)
