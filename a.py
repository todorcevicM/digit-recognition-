import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pylab as pl
import time 

from sklearn.model_selection import train_test_split, StratifiedKFold, LeaveOneOut
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay
from sklearn.neural_network import MLPClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier


#podaci su vec uredjeni
df = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")

df.groupby("label").count().iloc[:,:1]

def reduce_img(img):
    test = np.array(img).reshape(28,28)
    out = np.zeros(shape=(14,14))
    for i in range(14):
        for j in range(14):
            out[i,j]= ((test[i*2,j*2] + test[i*2+1,j*2] + test[i*2,j*2+1] + test[i*2+1,j*2+1]))

    return out

non_reduced = df.iloc[:,1:].copy()
x_train_non_cut = np.zeros(shape=(non_reduced.shape[0], 14*14))

for i in range(42000):
    tmp = non_reduced.iloc[i]
    x_train_non_cut[i] = reduce_img(tmp).reshape(1,14*14)


x_train = np.zeros(shape=(42000,10*10))
for i in range(42000):
    tmp = x_train_non_cut[i].reshape(14, 14)
    x_train[i] = tmp[2:12, 2:12].reshape(1,100)

y_train = df.iloc[:,0].copy()


def tacnost_po_klasi(mat_konf, klase):
    tacnost_i = []
    N = mat_konf.shape[0]
    for i in range(N):
        j = np.delete(np.array(range(N)),i) 
        TP = mat_konf[i,i]
        F = 0
        F = (sum(mat_konf[i,j]) + sum(mat_konf[j,i]))
        TN = sum(sum(mat_konf)) - F - TP
        tacnost_i.append((TP+TN)/sum(sum(mat_konf)))
        print('Za klasu ', klase[i], ' tacnost je: ', tacnost_i[i])
    tacnost_avg = np.mean(tacnost_i)
    return tacnost_avg

def osetljivost_po_klasi(mat_konf, klase):
    osetljivost_i = []
    N = mat_konf.shape[0]
    for i in range(N):
        j = np.delete(np.array(range(N)),i) 
        TP = mat_konf[i,i]
        FN = sum(mat_konf[i,j])
        osetljivost_i.append(TP/(TP+FN))
        print('Za klasu ', klase[i], ' osetljivost je: ', osetljivost_i[i])
    osetljivost_avg = np.mean(osetljivost_i)
    return osetljivost_avg

def specificnost_po_klasi(mat_konf, klase):
  specificnost_i = []
  N = mat_konf.shape[0]
  for i in range(N):
    j = np.delete(np.array(range(N)),i) 
    TP = mat_konf[i,i]
    FN = sum(mat_konf[i,j])
    F = (sum(mat_konf[i,j]) + sum(mat_konf[j,i]))
    TN = sum(sum(mat_konf)) - F - TP
    FP = F - FN
    specificnost_i.append(TN/(TN+FP))
    print('Za klasu ', klase[i], ' specificnost je: ', specificnost_i[i])
  return np.mean(specificnost_i)

def preciznost_po_klasi(mat_konf, klase):
  preciznost_i = []
  N = mat_konf.shape[0]
  for i in range(N):
    j = np.delete(np.array(range(N)),i) 
    TP = mat_konf[i,i]
    FN = sum(mat_konf[i,j])
    F = (sum(mat_konf[i,j]) + sum(mat_konf[j,i]))
    TN = sum(sum(mat_konf)) - F - TP
    FP = F - FN
    preciznost_i.append(TP/(TP+FP))
    print('Za klasu ', klase[i], ' preciznost je: ', preciznost_i[i])
  return np.mean(preciznost_i)


hidden_layer_sizes_bunch = [256]
activation_bunch = ["relu"]
solver_bunch = ["adam"]

time_start = time.time()
metric_banch = ["manhattan"]
for n in range(10):
  print("######")
  print(f"komsije {n+1}")
  for metric_fun in metric_banch:
    print("##########")
    print(f"metric_fun {metric_fun}")
    kf = StratifiedKFold(n_splits= 5, shuffle=True, random_state=42)
    indexes = kf.split(x_train, y_train)
    fin_conf_mat = np.zeros((len(np.unique(y_train)),len(np.unique(y_train))))
    for train_index, test_index in indexes:
        classifier = KNeighborsClassifier(n_neighbors= n+1, metric=metric_fun)
        classifier.fit(x_train[train_index], y_train[train_index])
        y_pred = classifier.predict(x_train[test_index])
        fin_conf_mat += confusion_matrix(y_train[test_index], y_pred)
        plt.show()
        print(accuracy_score(y_train[test_index], y_pred))

    print('procenat tacno predvidjenih: ', sum(np.diag(fin_conf_mat))/sum(sum(fin_conf_mat)))    
    print('finalna matrica je: ')
    disp = ConfusionMatrixDisplay(confusion_matrix =fin_conf_mat,  display_labels=classifier.classes_)

    cm = confusion_matrix(np.arange(25), np.arange(25))
    fig, ax = plt.subplots(figsize=(10,10))
    disp.plot(ax=ax, cmap="Blues", values_format='')

    plt.show()

print(f"time taken: {time.time() - time_start}")