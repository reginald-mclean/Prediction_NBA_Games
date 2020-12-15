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
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

data01 = np.genfromtxt("normalized_15_game_average_final.csv", delimiter=',', dtype=None,
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
        X_test = X_train
        y_test = y_train
        X_test = X_valid
        y_test = y_valid

        treee = RandomForestClassifier(bootstrap=False, max_leaf_nodes=100, n_estimators=1200, max_depth=90,
                                       max_features='auto', min_samples_leaf=1, min_samples_split=2,
                                       random_state=None)
        treee.fit(X_train, y_train)
        y_pred = treee.predict(X_test)

        acc = accuracy(y_test, np.transpose(y_pred))
        accc[it-1] = acc
        #print("Accuracy:", acc)

    print(accc)

trials(X,y)