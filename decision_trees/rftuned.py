import math
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from pprint import pprint
from sklearn.model_selection import RandomizedSearchCV
import os
import pandas as pd
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

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

data01 = np.genfromtxt("normalized_5_game_average_final.csv", delimiter=',', dtype=None,
                     encoding=None, names=True)
df = pd.DataFrame(data01)
print(df)

features = df[['HteamFG', 'HteamTS', 'HteamEFG', 'HteamPPS', 'HteamFIC', 'HteamFIC40', 'HteamOrtg', 'HteamEDiff',
                   'HteamPlay', 'Hwin', 'AteamTS', 'AteamFIC', 'AteamFIC40', 'AteamEDiff', 'Awin']]
target = df[['target']]

dataset = features.to_numpy()
target = target.to_numpy()

X = np.array(dataset[:,:-1], dtype=np.float64)
y = np.array(target[:, -1], dtype=np.int64)

X_train = X[:6148,:]
#X_train = X[7000:,:]
X_test = X[6149:,:]
y_train = y[:6148,]
#y_train = y[7000:,]
y_test = y[6149:,]

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
#print(y_test)

#classifier = RandomForestClassifier(n_estimators = 1000, random_state = 0)
#classifier.fit(X_train, y_train)
#y_pred = classifier.predict(X_test)

#acc = accuracy(y_test, np.transpose(y_pred))
#print()
#print("Accuracy:", acc)
print()

rf = RandomForestClassifier(random_state = 42)
print('Parameters currently in use:\n')
pprint(rf.get_params())
print()

n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
max_leaf_nodes= [25,100,200]
bootstrap = [True, False]
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'max_leaf_nodes': max_leaf_nodes,
               'bootstrap': bootstrap}
pprint(random_grid)

rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2,
                              random_state=42, n_jobs = -1)

rf_random.fit(X_train, y_train)

pprint(rf_random.best_params_)


def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    #predictions = np.transpose(predictions)
    accuracy = np.sum(test_labels == predictions) / len(test_labels)
    print()
    print('Model Performance')
    print('Accuracy = {:0.2f}%.'.format(accuracy))

    return accuracy

best_random = rf_random.best_estimator_
random_accuracy = evaluate(best_random, X_test, y_test)


print()

param_grid = {
    'bootstrap': [False],
    'max_leaf_nodes': [100,200,500],
    'max_depth': [80, 90, 100, 110],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200, 300, 1000]
}

rf = RandomForestClassifier()

grid_search = GridSearchCV(estimator = rf, param_grid = param_grid,
                          cv = 3, n_jobs = -1, verbose = 2)

grid_search.fit(X_train, y_train)
pprint(grid_search.best_params_)

best_grid = grid_search.best_estimator_
grid_accuracy = evaluate(best_grid, X_test, y_test)