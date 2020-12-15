import pandas as pd
import xgboost as kgbeast
import optuna

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def main(filename, num_epochs=1000, reg=False, fout="default"):
    # Reading in data
    data = pd.read_csv(filename, index_col=[0])
    y = data['target'].copy()
    X = data.drop('target', axis=1).copy()

    X_training, X_test, y_training, y_test = train_test_split(X, y, train_size=0.7, random_state=42)
    X_train, X_Valid, y_train, y_valid = train_test_split(X_training, y_training, train_size=0.9, random_state=500)

    # Creating d-matrix for XGBoost
    dtrain = kgbeast.DMatrix(X_train, label=y_train)
    dvalid = kgbeast.DMatrix(X_Valid, label=y_valid)
    dtest = kgbeast.DMatrix(X_test, label=y_test)

    # Hyperparameter Selection for XGBoost using Optuna
    def objective(trial):
        # Suggesting hyperparameter values for Optuna to try
        learning_rate = trial.suggest_loguniform('eta', 0.00001, 1)
        max_depth = trial.suggest_int('max_depth', 4, 8)
        gamma = trial.suggest_uniform('gamma', 0, 1)
        min_child_weight = trial.suggest_int('min_child_weight', 1, 10)

        # Creating hyperparameter dictionary for XGBoost
        hyperparam = {
            'max_depth': max_depth,
            'eta': learning_rate,
            'alpha': 0,
            'lambda': 0,
            'gamma': gamma,
            'min_child_weight': min_child_weight,
            'objective': 'multi:softmax',
            'num_class': 2}
        epochs = trial.suggest_categorical('epochs', [100, 300, 500, 700, 1000])

        # Implementing Regularization
        if reg:
            l1_reg = trial.suggest_loguniform('alpha', 0.00001, 10.0)
            l2_reg = trial.suggest_loguniform('lambda', 0.00001, 10.0)
            hyperparam["alpha"] = l1_reg
            hyperparam["lambda"] = l2_reg

        # Training and testing model
        model = kgbeast.train(hyperparam, dtrain, epochs)
        y_pred = model.predict(dvalid)

        # Accuracy check
        accuracy = accuracy_score(y_valid, y_pred)
        return accuracy

    # optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=1000)
    print('This optimization was performed with:', num_epochs, 'epochs, regularization =', reg, 'and file:', filename)
    print('The best hyperparameter values are:', study.best_params)
    print('The highest model accuracy was:', study.best_value)

    # Outputting results to textfile
    if fout != "default":
        fo = open(fout, "w")
        fo.write('This optimization was performed with: ' + str(num_epochs) + ' epochs, regularization = ' + str(reg) + ' and file: ' + filename + '\n\n')
        fo.write('The highest model accuracy was: ' + str(study.best_value) + '\n\n')
        fo.write('The best hyperparameter values are as follows:' + '\n\n')
        for k, v in study.best_params.items():
            fo.write(str(k) + ' >>> ' + str(v) + '\n\n')
        fo.close()


if __name__ == '__main__':
    main('normalized_5_game_average_final.csv', fout="hyperparam_norm_5_noreg.txt")
    main('normalized_10_game_average_final.csv', fout="hyperparam_norm_10_noreg.txt")
    main('normalized_15_game_average_final.csv', fout="hyperparam_norm_15_noreg.txt")
    main('unnormalized_5_game_average_final.csv', fout="hyperparam_unnorm_5_noreg.txt")
    main('unnormalized_10_game_average_final.csv', fout="hyperparam_unnorm_10_noreg.txt")
    main('unnormalized_15_game_average_final.csv', fout="hyperparam_unnorm_15_noreg.txt")
    main('normalized_5_game_average_final.csv', reg=True, fout="hyperparam_norm_5_reg.txt")
    main('normalized_15_game_average_final.csv', reg=True, fout="hyperparam_norm_15_reg.txt")
    main('normalized_10_game_average_final.csv', reg=True, fout="hyperparam_norm_10_reg.txt")
    main('unnormalized_5_game_average_final.csv', reg=True, fout="hyperparam_unnorm_5_reg.txt")
    main('unnormalized_10_game_average_final.csv', reg=True, fout="hyperparam_unnorm_10_reg.txt")
    main('unnormalized_15_game_average_final.csv', reg=True, fout="hyperparam_unnorm_15_reg.txt")



