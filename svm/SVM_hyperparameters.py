from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import optuna

def main(file_path, eval_kernel='rbf'):
    data = pd.read_csv(file_path, index_col=[0])
    y = data['target'].copy()
    x = data.drop('target', axis=1).copy()

    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=42)
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, train_size=0.9)

    x_train = x_train.to_numpy()
    y_train = y_train.to_numpy()

    x_valid = x_valid.to_numpy()
    y_valid = y_valid.to_numpy()

    x_test = x_test.to_numpy()
    y_test = y_test.to_numpy()

    def objective(trial):
        # suggest hyperparameters for Optuna to try
        gamma = trial.suggest_loguniform('gamma', 1e-5, 100)
        c_const = trial.suggest_loguniform('C', 1e-5, 100)
        model = svm.SVC(gamma=gamma, C=c_const, kernel=eval_kernel)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_valid)
        accuracy = accuracy_score(y_valid, y_pred)
        model = None
        if not model:
            print('continue')
        return accuracy*-1

    study = optuna.create_study()
    study.optimize(objective, n_trials=1000)
    print('file name:', file_path)
    print('Best hyperparameters:', study.best_params)
    print('Best accuracy:', study.best_value)
    best_params = study.best_params
    best_acc = study.best_value
    study = None
    return best_params, best_acc

if __name__ == '__main__':
    files = ['./Data/normalized_5_game_average_final.csv',
             './Data/normalized_10_game_average_final.csv',
             './Data/normalized_15_game_average_final.csv']
             # './Data/unnormalized_5_game_average_final.csv',
             # './Data/unnormalized_10_game_average_final.csv',
             # './Data/unnormalized_15_game_average_final.csv']
    accuracies = []
    params_summary = pd.DataFrame()

    for file in files:
        tuned_parameters, tuned_accuracy = main(file)
        params_summary = params_summary.append(tuned_parameters, ignore_index=True)
        accuracies.append(tuned_accuracy)
    accuracies = np.array(accuracies)
    params_summary['accuracy'] = accuracies*-1
    params_summary.insert(0, 'file', files)
    params_summary.to_csv('best_hyperparameters_rbf_kernel.csv')

    """
        testing linear kernel
    """

    accuracies = []
    params_summary = pd.DataFrame()

    for file in files:
        tuned_parameters, tuned_accuracy = main(file, eval_kernel='linear')
        params_summary = params_summary.append(tuned_parameters, ignore_index=True)
        accuracies.append(tuned_accuracy)
    accuracies = np.array(accuracies)
    params_summary['accuracy'] = accuracies*-1
    params_summary.insert(0, 'file', files)
    params_summary.to_csv('best_hyperparameters_linear_kernel.csv')

    """
        testing sigmoid kernel
    """

    accuracies = []
    params_summary = pd.DataFrame()

    for file in files:
        tuned_parameters, tuned_accuracy = main(file, eval_kernel='sigmoid')
        params_summary = params_summary.append(tuned_parameters, ignore_index=True)
        accuracies.append(tuned_accuracy)
    accuracies = np.array(accuracies)
    params_summary['accuracy'] = accuracies*-1
    params_summary.insert(0, 'file', files)
    params_summary.to_csv('best_hyperparameters_sigmoid_kernel.csv')

    """
        testing poly deg 3 kernel
    """

    # accuracies = []
    # params_summary = pd.DataFrame()
    #
    # for file in files:
    #     tuned_parameters, tuned_accuracy = main(file, eval_kernel='poly')
    #     params_summary = params_summary.append(tuned_parameters, ignore_index=True)
    #     accuracies.append(tuned_accuracy)
    # accuracies = np.array(accuracies)
    # params_summary['accuracy'] = accuracies*-1
    # params_summary.insert(0, 'file', files)
    # params_summary.to_csv('best_hyperparameters_poly_kernel.csv')