import pandas as pd
import numpy as np
import xgboost as kgbeast

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def main(filename, num_trials=10, num_epochs=1000, params_list=None, reg=False, fout="XGBoost Results.txt", verbose=True):
    # Reading in data
    data = pd.read_csv(filename, index_col=[0])
    y = data['target'].copy()
    X = data.drop('target', axis=1).copy()

    X_training, X_test, y_training, y_test = train_test_split(X, y, train_size=0.7, random_state=42)

    # Hyperparameter Setup
    epochs = num_epochs
    if params_list is None:
        params_list = {
            'eta': 0.0015301109389079433,
            'max_depth': 6,
            'gamma': 0.07351644552669655,
            'min_child_weight': 2,
            'objective': 'multi:softmax',
            'num_class': 2}
        if reg:
            params_list["alpha"] = 4.786843830713312
            params_list["lambda"] = 1.2671051384146887

    metric_train = np.zeros([num_trials, 4])
    metric_valid = np.zeros([num_trials, 4])
    metric_test = np.zeros([num_trials, 4])

    # Training and testing model
    for i in range(num_trials):
        # Splitting train and valid data
        X_train, X_valid, y_train, y_valid = train_test_split(X_training, y_training, train_size=0.9)

        # Creating d-matrix for XGBoost
        dtrain = kgbeast.DMatrix(X_train, label=y_train)
        dvalid = kgbeast.DMatrix(X_valid, label=y_valid)
        dtest = kgbeast.DMatrix(X_test, label=y_test)

        print("Starting Trial:", i)
        model = kgbeast.train(params_list, dtrain, epochs)
        print('Training Complete')
        y_pred_train = model.predict(dtrain)
        y_pred_valid = model.predict(dvalid)
        y_pred_test = model.predict(dtest)
        print('Predictions Complete')

        # Calculating performance metrics
        # Accuracy
        metric_train[i, 0] = accuracy_score(y_train, y_pred_train)
        metric_valid[i, 0] = accuracy_score(y_valid, y_pred_valid)
        metric_test[i, 0] = accuracy_score(y_test, y_pred_test)
        # Precision
        metric_train[i, 1] = precision_score(y_train, y_pred_train)
        metric_valid[i, 1] = precision_score(y_valid, y_pred_valid)
        metric_test[i, 1] = precision_score(y_test, y_pred_test)
        # Recall
        metric_train[i, 2] = recall_score(y_train, y_pred_train)
        metric_valid[i, 2] = recall_score(y_valid, y_pred_valid)
        metric_test[i, 2] = recall_score(y_test, y_pred_test)
        # F1-Score
        metric_train[i, 3] = f1_score(y_train, y_pred_train)
        metric_valid[i, 3] = f1_score(y_valid, y_pred_valid)
        metric_test[i, 3] = f1_score(y_test, y_pred_test)

    # Calculating average metrics and std
    avg_metric_train = metric_train.mean(0)
    avg_metric_valid = metric_valid.mean(0)
    avg_metric_test = metric_test.mean(0)

    std_metric_train = metric_train.std(0)
    std_metric_valid = metric_valid.std(0)
    std_metric_test = metric_test.std(0)

    if verbose:
        print('This model has an average training accuracy of ', avg_metric_train[0], 'and a standard deviation of',
              std_metric_train[0])
        print('This model has an average training precision of ', avg_metric_train[1], 'and a standard deviation of',
              std_metric_train[1])
        print('This model has an average training recall of ', avg_metric_train[2], 'and a standard deviation of',
              std_metric_train[2])
        print('This model has an average training F1-Score of ', avg_metric_train[3], 'and a standard deviation of',
              std_metric_train[3])
        # print(metric_train)
        print('This model has an average validation accuracy of ', avg_metric_valid[0], 'and a standard deviation of',
              std_metric_valid[0])
        print('This model has an average validation precision of ', avg_metric_valid[1], 'and a standard deviation of',
              std_metric_valid[1])
        print('This model has an average validation recall of ', avg_metric_valid[2], 'and a standard deviation of',
              std_metric_valid[2])
        print('This model has an average validation F1-Score of ', avg_metric_valid[3], 'and a standard deviation of',
              std_metric_valid[3])
        # print(metric_valid)
        print('This model has an average test accuracy of ', avg_metric_test[0], 'and a standard deviation of',
              std_metric_test[0])
        print('This model has an average test precision of ', avg_metric_test[1], 'and a standard deviation of',
              std_metric_test[1])
        print('This model has an average test recall of ', avg_metric_test[2], 'and a standard deviation of',
              std_metric_test[2])
        print('This model has an average test F1-Score of ', avg_metric_test[3], 'and a standard deviation of',
              std_metric_test[3])
        # print(metric_test)

    if fout != "default":
        fo = open(fout, "a")
        fo.write('Model: ' + str(num_epochs) + ' epochs, regularization = ' + str(reg) +
                 ' and file: ' + filename + '\n\n')
        fo.write('Model Metrics: [Accuracy, Precision, Recall, F1]: ' + '\n\n')
        fo.write('Training Avg: ' + str(avg_metric_train) + '\n\n')
        fo.write('Training Std: ' + str(std_metric_train) + '\n\n')
        fo.write('Valid Avg: ' + str(avg_metric_valid) + '\n\n')
        fo.write('Valid Std: ' + str(std_metric_valid) + '\n\n')
        fo.write('Test Avg: ' + str(avg_metric_test) + '\n\n')
        fo.write('Test Std: ' + str(std_metric_test) + '\n\n')
        fo.close()


if __name__ == '__main__':
    # Single testruns
    # main('normalized_15_game_average_final.csv', reg=True, num_epochs=1500, fout="default",
    #                                                                     params_list={'eta': 0.122909965730246,
    #                                                                     'max_depth': 5, 'gamma': 0.13813736643829339,
    #                                                                     'min_child_weight': 7,
    #                                                                     'alpha': 9.958175166494591,
    #                                                                     'lambda': 0.00014143118115752048,
    #                                                                     'objective': 'multi:softmax', 'num_class': 2})

    # Normalized Datasets
    main('normalized_5_game_average_final.csv', params_list={'eta': 0.004959821756600845, 'max_depth': 4,
                                                             'gamma': 0.15788824897759618, 'min_child_weight': 10,
                                                             'objective': 'multi:softmax', 'num_class': 2})
    main('normalized_5_game_average_final.csv', reg=True, params_list={'eta': 0.00036511866209319483, 'max_depth': 7,
                                                                       'gamma': 0.7250540008189499, 'min_child_weight': 5,
                                                                       'objective': 'multi:softmax', 'num_class': 2,
                                                                       'alpha': 9.944302416347949,
                                                                       'lambda': 0.0001834874769790402})
    main('normalized_10_game_average_final.csv', params_list={'eta': 0.000203530215581263, 'max_depth': 4,
                                                              'gamma': 0.7666153201480636, 'min_child_weight': 7,
                                                              'objective': 'multi:softmax', 'num_class': 2})
    main('normalized_10_game_average_final.csv', reg=True, params_list={'eta': 5.783909755117914e-05, 'max_depth': 4,
                                                                        'gamma': 0.3998942473523745,
                                                                        'min_child_weight': 8,
                                                                        'objective': 'multi:softmax', 'num_class': 2,
                                                                        'alpha': 0.014531596443990732,
                                                                        'lambda': 0.0032249321534895156})
    main('normalized_15_game_average_final.csv', params_list={'eta': 0.0011477361053320362, 'max_depth': 7,
                                                              'gamma': 0.9775731439665086, 'min_child_weight': 10,
                                                              'objective': 'multi:softmax', 'num_class': 2})
    main('normalized_15_game_average_final.csv', reg=True, params_list={'eta': 0.0015301109389079433, 'max_depth': 6,
                                                                        'gamma': 0.07351644552669655,
                                                                        'min_child_weight': 2,
                                                                        'objective': 'multi:softmax', 'num_class': 2,
                                                                        'alpha': 4.786843830713312,
                                                                        'lambda': 1.2671051384146887})
    # Unnormalized Datasets
    main('unnormalized_5_game_average_final.csv', params_list={'eta': 0.0003859340250313295, 'max_depth': 6,
                                                               'gamma': 0.19893758821228094, 'min_child_weight': 3,
                                                               'objective': 'multi:softmax', 'num_class': 2})
    main('unnormalized_5_game_average_final.csv', reg=True, params_list={'eta': 6.493757606715137e-05, 'max_depth': 7,
                                                                         'gamma': 0.47316256369522525,
                                                                         'min_child_weight': 8,
                                                                         'objective': 'multi:softmax', 'num_class': 2,
                                                                         'alpha': 8.762080843284313e-05,
                                                                         'lambda': 0.1924926818894256})
    main('unnormalized_10_game_average_final.csv', params_list={'eta': 0.0006082349828325408, 'max_depth': 7,
                                                                'gamma': 0.002367672652523468, 'min_child_weight': 8,
                                                                'objective': 'multi:softmax', 'num_class': 2})
    main('unnormalized_10_game_average_final.csv', reg=True, params_list={'eta': 0.0007749766557281258, 'max_depth': 7,
                                                                          'gamma': 0.18291986752687792,
                                                                          'min_child_weight': 5,
                                                                          'objective': 'multi:softmax', 'num_class': 2,
                                                                          'alpha': 2.5723430322553593,
                                                                          'lambda': 0.15116101377009303})
    main('unnormalized_15_game_average_final.csv', params_list={'eta': 9.32861234621325e-05, 'max_depth': 5,
                                                                'gamma': 0.7071396040708907, 'min_child_weight': 8,
                                                                'objective': 'multi:softmax', 'num_class': 2})
    main('unnormalized_15_game_average_final.csv', reg=True, params_list={'eta': 0.0005729040733488505, 'max_depth': 5,
                                                                          'gamma': 0.45296164664950184,
                                                                          'min_child_weight': 4,
                                                                          'objective': 'multi:softmax', 'num_class': 2,
                                                                          'alpha': 0.0021500423969490067,
                                                                          'lambda': 0.0021938937302783054})