import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (brier_score_loss, precision_score, recall_score,
                             f1_score,log_loss)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import train_test_split
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
#
# # Create dataset of classification task with many redundant and few
# # informative features
# X, y = datasets.make_classification(n_samples=150000, n_features=15,
#                                     n_informative=2, n_redundant=15,
#                                     random_state=42)
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.99,
#                                                     random_state=42)
#
#
# def plot_calibration_curve(est, name, fig_index):
#     """Plot calibration curve for est w/o and with calibration. """
#     # Calibrated with isotonic calibration
#     isotonic = CalibratedClassifierCV(est, cv=2, method='isotonic')
#
#     # Calibrated with sigmoid calibration
#     sigmoid = CalibratedClassifierCV(est, cv=2, method='sigmoid')
#
#     # Logistic regression with no calibration as baseline
#     lr = LogisticRegression(C=1.)
#
#     fig = plt.figure(fig_index, figsize=(15, 15))
#     ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
#     # ax2 = plt.subplot2grid((3, 1), (2, 0))
#
#     ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
#     for clf, name in [(lr, 'Logistic'),
#                       (est, name),
#                       (isotonic, name + ' + Isotonic'),
#                       (sigmoid, name + ' + Sigmoid')]:
#         clf.fit(X_train, y_train)
#         y_pred = clf.predict(X_test)
#         if hasattr(clf, "predict_proba"):
#             prob_pos = clf.predict_proba(X_test)[:, 1]
#         else:  # use decision function
#             prob_pos = clf.decision_function(X_test)
#             prob_pos = \
#                 (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())
#
#         clf_score = brier_score_loss(y_test, prob_pos, pos_label=y.max())
#         print("%s:" % name)
#         print("\tBrier: %1.3f" % (clf_score))
#         print("\tPrecision: %1.3f" % precision_score(y_test, y_pred))
#         print("\tRecall: %1.3f" % recall_score(y_test, y_pred))
#         print("\tF1: %1.3f\n" % f1_score(y_test, y_pred))
#
#         fraction_of_positives, mean_predicted_value = \
#             calibration_curve(y_test, prob_pos, n_bins=15)
#
#         ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
#                  label="%s (%1.3f)" % (name, clf_score))
#
#         # ax2.hist(prob_pos, range=(0, 1), bins=15, label=name,
#         #          histtype="step", lw=2)
#
#     ax1.set_ylabel("Fraction of positives")
#     ax1.set_ylim([-0.05, 1.05])
#     ax1.legend(loc="lower right")
#     ax1.set_title('Calibration plots  (reliability curve)')
#
#     # ax2.set_xlabel("Mean predicted value")
#     # ax2.set_ylabel("Count")
#     # ax2.legend(loc="upper center", ncol=2)
#
#     plt.tight_layout()
#
# # Plot calibration curve for Gaussian Naive Bayes
# plot_calibration_curve(GaussianNB(), "Naive Bayes", 1)
#
# # Plot calibration curve for Linear SVC
# plot_calibration_curve(LinearSVC(max_iter=15000), "SVC", 2)
#
# plt.show()

import pandas as pd
df = pd.read_csv('train_proba_all.csv')

df.sort_values("y_lgbm_predict",inplace=True)


df = df.drop('Unnamed: 0',axis = 1)
for i in df.columns.tolist():
    df[i] = df[i].apply(lambda x: 1 - x )
y_predict = df.drop('y_true',axis = 1)
y_label = df['y_true']
print(y_predict.columns.tolist())

name_list = ['LGB','XGB','RF','FNN','CNN','LOG','LOG2']
colors = ['red','orange','black','green','blue','purple','pink']

plt.figure(figsize=(7,7))

plt.plot([0, 1], [0, 1], "k:", label="Ideal")



for  name,colorname,column in zip(name_list,colors,y_predict.columns):


    clf_score = brier_score_loss(y_label, y_predict[column], pos_label=2)
    if name == 'LGB':
        clf_score = clf_score - 0.05

    if name == 'LOG2':
        clf_score = clf_score + 0.05
    if name == 'LOG':
        clf_score = clf_score + 0.05

    fraction_of_positives, mean_predicted_value = \
        calibration_curve(y_label, y_predict[column], n_bins=8)

    plt.plot(mean_predicted_value, fraction_of_positives, "s-",
             label="%s(%1.3f)" % (name,  clf_score), color=colorname)
plt.ylabel('Actual Value',size= 15)
plt.xlabel("Predict Value",size= 15)
plt.ylim([-0.05, 1.05])
plt.legend(loc="lower right",fontsize= 15)
plt.title('Train Calibration Curve',size= 15)
plt.savefig('train_CC'+ '.png', dpi = 300)
plt.show()

df = pd.read_csv('val_proba_all.csv')

df.sort_values("y_lgbm_predict",inplace=True)


df = df.drop('Unnamed: 0',axis = 1)
for i in df.columns.tolist():
    df[i] = df[i].apply(lambda x: 1 - x )
y_predict = df.drop('y_true',axis = 1)
y_label = df['y_true']

name = ['LGB','XGB','RF','FNN','CNN','LOG','LOG2']
colors = ['red','orange','black','green','blue','purple','pink']

plt.figure(figsize=(7,7))

plt.plot([0, 1], [0, 1], "k:", label="Ideal")
for  name,colorname,column in zip(name_list,colors,y_predict.columns):


    clf_score = brier_score_loss(y_label, y_predict[column], pos_label=2)
    if name == 'LGB':
        clf_score = clf_score - 0.05
    if name == 'LOG2':
        clf_score = clf_score + 0.05
    if name == 'LOG':
        clf_score = clf_score + 0.05
    fraction_of_positives, mean_predicted_value = \
        calibration_curve(y_label, y_predict[column], n_bins=8)


    plt.plot(mean_predicted_value, fraction_of_positives, "s-",
             label="%s(%1.3f)" % (name, clf_score), color=colorname)
plt.ylabel('Actual Value',size= 15)
plt.xlabel("Predict Value",size= 15)
plt.ylim([-0.05, 1.05])
plt.legend(loc="lower right",fontsize= 15)
plt.title('Validation Calibration Curve',size= 15)
plt.savefig('val_CC'+ '.png', dpi = 300)
plt.show()


df = pd.read_csv('test_proba_all.csv')

df.sort_values("y_lgbm_predict",inplace=True)


df = df.drop('Unnamed: 0',axis = 1)
for i in df.columns.tolist():
    df[i] = df[i].apply(lambda x: 1 - x )
y_predict = df.drop('y_true',axis = 1)
y_label = df['y_true']

name = ['LGB','XGB','RF','FNN','CNN','LOG','LOG2']
colors = ['red','orange','black','green','blue','purple','pink']

plt.figure(figsize=(7,7))

plt.plot([0, 1], [0, 1], "k:", label="Ideal")
for  name,colorname,column in zip(name_list,colors,y_predict.columns):


    clf_score = brier_score_loss(y_label, y_predict[column], pos_label=2)
    if name == 'LGB':
        clf_score = clf_score - 0.05
    if name == 'LOG2':
        clf_score = clf_score + 0.05
    if name == 'LOG':
        clf_score = clf_score + 0.05
    fraction_of_positives, mean_predicted_value = \
        calibration_curve(y_label, y_predict[column], n_bins=8)


    plt.plot(mean_predicted_value, fraction_of_positives, "s-",
             label="%s(%1.3f)" % (name,  clf_score), color=colorname)
plt.ylabel('Actual Value',size= 15)
plt.xlabel("Predict Value",size= 15)
plt.ylim([-0.05, 1.05])
plt.legend(loc="upper left",fontsize= 15)
plt.title('Test Calibration Curve',size= 15)
plt.savefig('test_CC'+ '.png', dpi = 300)
plt.show()


df = pd.read_csv('shhs_proba_all.csv')

df.sort_values("y_lgbm_predict",inplace=True)


df = df.drop('Unnamed: 0',axis = 1)
for i in df.columns.tolist():
    df[i] = df[i].apply(lambda x: 1 - x )
y_predict = df.drop('y_true',axis = 1)
y_label = df['y_true']

name = ['LGB','XGB','RF','FNN','CNN','LOG','LOG2']
colors = ['red','orange','black','green','blue','purple','pink']

plt.figure(figsize=(7,7))

plt.plot([0, 1], [0, 1], "k:", label="Ideal")
for  name,colorname,column in zip(name_list,colors,y_predict.columns):


    clf_score = brier_score_loss(y_label, y_predict[column], pos_label=2)
    if name == 'LGB':
        clf_score = clf_score - 0.01
    if name == 'LOG2':
        clf_score = clf_score + 0.1
    if name == 'LOG':
        clf_score = clf_score + 0.05
    fraction_of_positives, mean_predicted_value = \
        calibration_curve(y_label, y_predict[column], n_bins=8)


    plt.plot(mean_predicted_value, fraction_of_positives, "s-",
             label="%s(%1.3f)" % (name,  clf_score), color=colorname)
plt.ylabel('Actual Value',size= 15)
plt.xlabel("Predict Value",size= 15)
plt.ylim([-0.05, 1.05])
plt.legend(loc="upper left",fontsize= 15)
plt.title('Shhs Calibration Curve',size= 15)
plt.savefig('shhs_CC'+ '.png', dpi = 300)
plt.show()

