import math
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from pprint import pprint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import validation_curve
import os
import pandas as pd
from mlxtend.evaluate import bias_variance_decomp
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from collections import Counter
from random import seed
from math import sqrt
from random import randrange
import graphviz
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor

data01 = np.genfromtxt("unnormalized_15_game_average_final.csv", delimiter=',', dtype=None,
                     encoding=None, names=True)
df = pd.DataFrame(data01)

features = df[['HteamFG', 'HteamTS', 'HteamEFG', 'HteamPPS', 'HteamFIC', 'HteamFIC40', 'HteamOrtg', 'HteamEDiff',
                   'HteamPlay', 'Hwin', 'AteamTS', 'AteamFIC', 'AteamFIC40', 'AteamEDiff', 'Awin']]
target = df[['target']]

dataset = features.to_numpy()
target = target.to_numpy()

X = np.array(dataset[:,:-1], dtype=np.float64)
y = np.array(target[:, -1], dtype=np.int64)

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

def trials(x,Y):
    accc = np.zeros((10,1))
    ebv = np.zeros((10,3))
    prf = np.zeros((10,3))

    for it in np.arange(10):
        X_train, X_test, y_train, y_test = train_test_split(x, Y, train_size=0.7, random_state=42)
        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, train_size=0.9)
        #X_test = X_train
        #y_test = y_train
        #X_test = X_valid
        #y_test = y_valid

        treee = DecisionTreeClassifier(max_depth=4, min_samples_split=500,
                                       min_samples_leaf=450, max_features=None, random_state=None, max_leaf_nodes=10)
        treee.fit(X_train, y_train)
        y_pred = treee.predict(X_test)

        acc = accuracy(y_test, np.transpose(y_pred))
        accc[it-1] = acc
        #print("Accuracy:", acc)

        mse, bias, var = bias_variance_decomp(treee, X_train, y_train, X_test, y_test, loss='0-1_loss', random_seed=123)
        ebv[it - 1,0:3] = mse, bias, var
        #print()
        #print('Average Expected Loss: %.3f' % mse)
        #print('Bias: %.3f' % bias)
        #print('Variance: %.3f' % var)

        p = precision_score(y_test, y_pred, average='binary')
        r = recall_score(y_test, y_pred, average='binary')
        f = f1_score(y_test, y_pred, average='binary')
        prf[it - 1, 0:3] = p,r,f
        #print()
        #print('Precision: %.3f' % p)
        #print('Recall: %.3f' % r)
        #print('f1: %.3f' % f)
    print(accc)
    print()
    print(ebv)
    print()
    print(prf)

trials(X,y)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, train_size=0.9)

train_scores, test_scores = validation_curve(
            DecisionTreeClassifier(), X_train, y_train, param_name="min_samples_split",
            param_range=np.arange(2, 10, 1), cv=5, scoring="accuracy"
        )

train_scores_mean = np.mean(train_scores,axis=1)
train_scores_std = np.std(train_scores,axis=1)
test_scores_mean = np.mean(test_scores,axis=1)
test_scores_std = np.std(test_scores,axis=1)
param_range=np.arange(2, 10,1)

plt.title("Validation Curve with Decision Trees")
plt.xlabel("min_samples_split")
plt.ylabel("Score")
plt.ylim(0.0, 1.1)
plt.plot(param_range, train_scores_mean, label="Training score",
             color="darkorange", lw=2)
plt.plot(param_range, test_scores_mean, label="Cross-validation score",
             color="navy", lw=2)
plt.legend(loc="best")
plt.show()

