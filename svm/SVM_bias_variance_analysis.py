from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import pandas as pd


def set_up_data(file_name):
    data = pd.read_csv(file_name, index_col=[0])
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

    return x_train, y_train, x_valid, y_valid, x_test, y_test


def bias_variance_analysis(file_name, gamma, C, kernel, num_trials=10):
    accuracy_train = []
    precision_train = []
    recall_train = []
    f1_train = []
    accuracy_valid = []
    precision_valid = []
    recall_valid = []
    f1_valid = []
    accuracy_test = []
    precision_test = []
    recall_test = []
    f1_test = []
    print(file_name)
    verbose_df = pd.DataFrame()
    for trial in range(num_trials):
        """
        model training
        """
        x_train, y_train, x_valid, y_valid, x_test, y_test = set_up_data(file_name)
        model = svm.SVC(gamma=gamma, C=C, kernel=kernel)
        model.fit(x_train, y_train)
        """
        training set
        """
        y_train_pred = model.predict(x_train)
        accuracy_train.append(accuracy_score(y_train, y_train_pred))
        precision_train.append(precision_score(y_train, y_train_pred))
        recall_train.append(recall_score(y_train, y_train_pred))
        f1_train.append(f1_score(y_train, y_train_pred))
        """
        validation set
        """
        y_valid_pred = model.predict(x_valid)
        accuracy_valid.append(accuracy_score(y_valid, y_valid_pred))
        precision_valid.append(precision_score(y_valid, y_valid_pred))
        recall_valid.append(recall_score(y_valid, y_valid_pred))
        f1_valid.append(f1_score(y_valid, y_valid_pred))
        """
        test set
        """
        y_test_pred = model.predict(x_test)
        accuracy_test.append(accuracy_score(y_test, y_test_pred))
        precision_test.append(precision_score(y_test, y_test_pred))
        recall_test.append(recall_score(y_test, y_test_pred))
        f1_test.append(f1_score(y_test, y_test_pred))
    """
    training set
    """
    verbose_df['training_accuracy'] = accuracy_train
    verbose_df['training_precision'] = precision_train
    verbose_df['training_recall'] = recall_train
    verbose_df['training_f1'] = f1_train
    """
    validation set
    """
    verbose_df['validation_accuracy'] = accuracy_valid
    verbose_df['validation_precision'] = precision_valid
    verbose_df['validation_recall'] = recall_valid
    verbose_df['validation_f1'] = f1_valid
    """
    test set
    """
    verbose_df['test_accuracy'] = accuracy_test
    verbose_df['test_precision'] = precision_test
    verbose_df['test_recall'] = recall_test
    verbose_df['test_f1'] = f1_test
    print(verbose_df.head())
    """
    accuracy
    """
    accuracy_train = np.array(accuracy_train)
    accuracy_valid = np.array(accuracy_valid)
    accuracy_test = np.array(accuracy_test)
    """
    precision
    """
    precision_train = np.array(precision_train)
    precision_valid = np.array(precision_valid)
    precision_test = np.array(precision_test)
    """
    recall
    """
    recall_train = np.array(recall_train)
    recall_valid = np.array(recall_valid)
    recall_test = np.array(recall_test)
    """
    f1 scores
    """
    f1_train = np.array(f1_train)
    f1_valid = np.array(f1_valid)
    f1_test = np.array(f1_test)
    """
    training set
    """
    train_mean = np.mean(accuracy_train)
    train_std = np.std(accuracy_train)
    train_prec = np.mean(precision_train)
    train_prec_std = np.std(precision_train)
    train_rec = np.mean(recall_train)
    train_rec_std = np.std(recall_train)
    train_f1 = np.mean(f1_train)
    train_f1_std = np.std(f1_train)
    """
    validation set
    """
    valid_mean = np.mean(accuracy_valid)
    valid_std = np.std(accuracy_valid)
    valid_prec = np.mean(precision_valid)
    valid_prec_std = np.std(precision_valid)
    valid_rec = np.mean(recall_valid)
    valid_rec_std = np.std(recall_valid)
    valid_f1 = np.mean(f1_valid)
    valid_f1_std = np.std(f1_valid)
    """
    test set
    """
    test_mean = np.mean(accuracy_test)
    test_std = np.std(accuracy_test)
    test_prec = np.mean(precision_test)
    test_prec_std = np.std(precision_test)
    test_rec = np.mean(recall_test)
    test_rec_std = np.std(recall_test)
    test_f1 = np.mean(f1_test)
    test_f1_std = np.std(f1_test)
    """
    organizing stuff
    """
    accuracy_columns = ['train_accuracy', 'train_accuracy_std',
                        'validation_accuracy', 'validation_accuracy_std',
                        'test_accuracy', 'test_accuracy_std']
    precision_columns = ['train_prec', 'train_prec_std',
                         'validation_prec', 'validation_prec_std',
                         'test_prec', 'test_prec_std']
    recall_columns = ['train_recall', 'train_recall_std',
                      'validation_recall', 'validation_recall_std',
                      'test_recall', 'test_recall_std']
    f1_columns = ['train_f1', 'train_f1_std',
                  'validation_f1', 'validation_f1_std',
                  'test_f1', 'test_f1_std']
    accuracy_df = pd.DataFrame([[train_mean, train_std, valid_mean, valid_std, test_mean, test_std]],
                               columns=accuracy_columns)
    precision_df = pd.DataFrame([[train_prec, train_prec_std, valid_prec, valid_prec_std, test_prec, test_prec_std]],
                                columns=precision_columns)
    recall_df = pd.DataFrame([[train_rec, train_rec_std, valid_rec, valid_rec_std, test_rec, test_rec_std]],
                             columns=recall_columns)
    f1_df = pd.DataFrame([[train_f1, train_f1_std, valid_f1, valid_f1_std, test_f1, test_f1_std]],
                         columns=f1_columns)
    data_output = pd.concat([accuracy_df, precision_df, recall_df, f1_df], axis=1)
    return data_output


def process_data(file_path, kernel, write_file):
    parameter_data = pd.read_csv(file_path)
    file_names = parameter_data['file'].to_numpy()
    reg_params = parameter_data['C'].to_numpy()
    gamma_params = parameter_data['gamma'].to_numpy()
    summary = pd.DataFrame()
    statistic_summary = pd.DataFrame()
    for index, read_file in enumerate(file_names):
        print('kernel:', kernel)
        stats_df = bias_variance_analysis(read_file, gamma_params[index], reg_params[index], kernel)
        if statistic_summary.empty:
            statistic_summary = pd.DataFrame(columns=stats_df.keys())
        statistic_summary = pd.concat([statistic_summary, stats_df], ignore_index=True)

    summary['file'] = file_names
    summary['C'] = reg_params
    summary['gamma'] = gamma_params
    summary = pd.concat([summary, statistic_summary], axis=1)
    print(summary.head())

    summary.to_csv(write_file)


if __name__ == '__main__':
    files = ['best_hyperparameters_linear_kernel.csv', 'best_hyperparameters_rbf_kernel.csv',
             'best_hyperparameters_sigmoid_kernel.csv']
    write_files = ['SVM_bias_variance_linear_kernel.csv', 'SVM_bias_variance_rbf_kernel.csv',
                   'SVM_bias_variance_sigmoid_kernel.csv']
    kernels = ['linear', 'rbf', 'sigmoid']

    for i, file in enumerate(files):
        process_data(file, kernels[i], write_files[i])
