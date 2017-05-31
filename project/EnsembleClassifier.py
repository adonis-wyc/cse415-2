'''
EnsembleClassifier.py
William Menten-Weil (wtmenten) & Graham Kelly (grahamtk)
This is the main file for our interactive Ensemble classifier
with K Fold cross validation and Bootstrap Aggregating

Running this file as main will give you an interactive prompt
'''
import csv
import random
import itertools
import sys, os, pickle
import numpy as np
import math
from os import path
import random
from collections import defaultdict
from sklearn import datasets
from DecisionTreeClassifier import DecisionTreeClassifierCustom
from NaiveBayesClassifier import GenericNB


iris = datasets.load_iris()
X = iris.data[:, :]  # we only take the first two features.
Y = iris.target
# print(X.T.shape)
# print(Y.T.shape)
dataset = np.concatenate([X.T, [Y.T]]).T
# print(dataset.shape)

def train_test_split(*args, ratio=0.2):
    assert len(args) > 0
    split_index = int(round(len(args[0]) * (1-ratio)))
    train = [arg[:split_index] if arg is not None else None for arg in args]
    test = [arg[split_index:] if arg is not None else None for arg in args]
    return train, test


def load_csv(filename):
    # loads the csv by given filename
    lines = list(csv.reader(open(filename, "r")))
    headers = lines[0]
    index = [line[0] for line in lines[1:]]
    dataset = [line[1:] for line in lines[1:]]
    for i in range(len(dataset)):
        dataset[i] = [float(x) for x in dataset[i]]
    return dataset

def split_dataset(dataset, splitRatio):
    # splits the dataset by the given ratio
    trainSize = int(len(dataset) * splitRatio)
    trainSet = []
    copy = list(dataset)
    while len(trainSet) < trainSize:
        index = random.randrange(len(copy))
        trainSet.append(copy.pop(index))
    return [trainSet, copy]

def gen_folds(df,n_splits=10):
    step = math.ceil(len(df) / n_splits)
    size = len(df)
    folds = []
    random_index = np.array(list(range(len(df))))
    random.shuffle(random_index)
    for loc in range(0,size,step):
        test_indexer = range(loc,size) if loc+step > size else range(loc,loc+step)
        train_indexer = list(itertools.chain(range(0,loc),range(loc+step,size)))
        folds.append([random_index[train_indexer],random_index[test_indexer]])
    return folds

def bootstrap_resample(X, n=None, n_datasets=100):
    if n == None:
        n = len(X)

    for i in range(n_datasets):
        resample_i = np.floor(np.random.rand(n)*len(X)).astype(int)
        X_resample = np.array(X[resample_i])
        yield X_resample

def get_ensemble_classification(ens, X):
    preds = [en.predict(X) for en in ens]
    rows = []
    for row in zip(*preds):
        row_counts = defaultdict(int)
        for i in range(2):
            row_counts[i] = 0

        for v in row:
            row_counts[v] += 1
        arg_counts = [x[1] for x in sorted(row_counts.items(), key=lambda x: x[0])]
        rows.append(arg_counts)
    classifications = [row.index(max(row)) for row in rows]
    return classifications

def accuracy_score(Y, predictions):
    correct = 0
    for i in range(len(Y)):
        if Y[i] == predictions[i]:
            correct += 1
    return (correct/float(len(Y)))


def map_continuous(f):
    if f < -0.01: return 0
    elif -0.01 < f < 0.01: return 1
    else: return 2

def bin_continuous(arr):
    return np.array([map_continuous(a) for a in arr])

def main(dataset_name="iris", model_name='both', verbose=2, folds=10, ensemble_size=9, bootstrap=True, save_name=None, load_name=None):
    probas = []
    iter_weights = []
    global dataset, load_fail
    splitRatio = 0.67
    dataset, X, Y = None, None, None
    if dataset_name == 'finance':
        filename = 'dataframe.csv'
        dataset = np.array(load_csv(filename))
        dataset = dataset[:200]
        # print(dataset.shape)
        X = dataset.T[1:].T
        Y = dataset.T[0].T
        Y = bin_continuous(Y)
    else:
        iris = datasets.load_iris()
        X = iris.data[:, :4]  # we only take the first 4 features.
        Y = iris.target
        dataset = np.concatenate([X.T, [Y.T]]).T


    ensemble_2 = np.array([])
    train_errs = []
    test_errs = []
    load_fail = True

    if load_name != None:
        load_fail = False
        file = path.dirname(path.abspath(__file__))
        file += '/saves/' + load_name + '.pickle'
        try:
            load_file = open(file, 'rb')
            ensemble_2 = pickle.load(load_file)
            if verbose > 0:
                print('Ensemble loaded.')
            if verbose > -1:
                random_index = np.array(list(range(len(X))))
                random.shuffle(random_index)
                split_index = math.floor(len(X) * 0.80)
                train_idx, test_idx = random_index[:split_index], random_index[split_index:]
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = Y[train_idx], Y[test_idx]
                test_preds = get_ensemble_classification(ensemble_2, X_test)
                test_acc = accuracy_score(y_test, test_preds)
                print("Cross-fold Ensemble Acc: ", test_acc)
        except FileNotFoundError as e:
            if verbose > -1:
                print('Failed to load ensemble from %s' % file)
                print(e)
            load_fail = True
    if load_fail:
        print('Constructing new ensemble')
        for fold_round, (train_idx, test_idx) in enumerate(gen_folds(X, n_splits=folds)):
            if verbose > 0:
                print('')
                print("Running Fold %s" % (fold_round+1))
            ensemble = []
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = Y[train_idx], Y[test_idx]

            # trains model on non bootstrapped data first
            if verbose > 1:
                print('')
            if model_name in ['decision tree', 'both']:
                clf = DecisionTreeClassifierCustom()
                clf.fit(X_train, y_train)
                ensemble.append(clf)
                if verbose > 1:
                    print("Trained Classifer %s with score %1.4f" % ('Decision Tree', clf.score(X_test, y_test)))

            if model_name in ['naive bayes', 'both']:
                nb = GenericNB()
                nb.fit(X_train,y_train)
                ensemble.append(nb)
                if verbose > 1:
                    print("Trained Classifer %s with score %1.4f" % ('Naive Bayes', nb.score(X_test, y_test)))

            if bootstrap == True:
                if verbose > 0:
                    print("Bootstrapping...")
                for boot_round, sample in enumerate(bootstrap_resample(np.asmatrix(X_train), n_datasets=ensemble_size)):
                    if verbose > 1:
                        print('')
                    if model_name in ['decision tree', 'both']:
                        clf = DecisionTreeClassifierCustom()
                        clf.fit(sample, y_train)
                        ensemble.append(clf)
                        if verbose > 1:
                            print("Trained Classifer %s with score %1.4f" % ('Decision Tree', clf.score(X_test, y_test)))

                    if model_name in ['naive bayes', 'both']:
                        nb = GenericNB()
                        nb.fit(sample,y_train)
                        ensemble.append(nb)
                        if verbose > 1:
                            print("Trained Classifer %s with score %1.4f" % ('Naive Bayes', nb.score(X_test, y_test)))


            train_preds = get_ensemble_classification(ensemble, X_train)
            test_preds = get_ensemble_classification(ensemble, X_test)
            train_acc = accuracy_score(y_train, train_preds)
            test_acc = accuracy_score(y_test, test_preds)
            if verbose > 0:
                print("Fold Ensemble Scores train: %1.4f - test: %1.4f" % (train_acc, test_acc))
            train_errs.append(train_acc)
            test_errs.append(test_acc)
            ensemble_2 = np.concatenate([ensemble_2, ensemble])

        if save_name != None:
            file = os.getcwd()
            file += '/saves/' + save_name + '.pickle'
            with open(file, 'wb') as save_file:
                save_file.write(pickle.dumps(ensemble_2))
            if verbose > 0:
                print('Ensemble saved.')

        if verbose > -1:
            test_preds = get_ensemble_classification(ensemble_2, X_test)
            test_acc = accuracy_score(y_test, test_preds)
            print("Cross-fold Ensemble Acc: ", test_acc)
            print('Avg. Ensemble accuracy for train data:', np.mean(train_errs))
            print('Avg. Ensemble accuracy for test data:', np.mean(test_errs))


def run_interactive():
    program_kwargs = dict(
        dataset_name = None,
        model_name = None,
        ensemble_size = None,
        folds = None,
        bootstrap = None,
        save_name = None,
        load_name = None,
        verbose = None
    )
    valid_datasets = ['iris', 'finance']
    while program_kwargs['dataset_name'] not in valid_datasets:
        program_kwargs['dataset_name'] = str(input("Which dataset would you like to use? ('finance' or 'iris'): ")).lower()
        if program_kwargs['dataset_name'] not in valid_datasets:
            print("Sorry I didn't recognize that dataset.")

    valid_models = ['naive bayes', 'decision tree', 'both']
    while program_kwargs['model_name'] not in valid_models:
        program_kwargs['model_name'] = str(input("Which model would you like to use? ('naive bayes', 'decision tree', 'both'): ")).lower()
        if program_kwargs['model_name'] not in valid_models:
            print("Sorry I didn't recognize that model.")


    valid_bag_opts = ['y', 'n']
    while program_kwargs['bootstrap'] not in valid_bag_opts:
        program_kwargs['bootstrap'] = str(input("Should we use Bootstrap Aggregating (Bagging)? (y/n): ")).lower()
        if program_kwargs['bootstrap'] not in valid_bag_opts:
            print("Sorry I didn't recognize that answer. (y/n)")
        else:
            program_kwargs['bootstrap'] = True if program_kwargs['bootstrap'] == 'y' else False
            break

    if program_kwargs['bootstrap']:
        while type(program_kwargs['ensemble_size']) != type(1):
            try:
                program_kwargs['ensemble_size'] = input("What size ensemble would you like to use? (int): ")
                program_kwargs['ensemble_size'] = int(program_kwargs['ensemble_size'])
            except:
                print("Sorry I didn't recognize that number.")
                program_kwargs['ensemble_size'] = None

    while type(program_kwargs['folds']) != type(1):
            try:
                program_kwargs['folds'] = input("How many folds would you like to use? (int): ")
                program_kwargs['folds'] = int(program_kwargs['folds'])
            except:
                print("Sorry I didn't recognize that number.")
                program_kwargs['folds'] = None

    save_opt, load_opt = None, None
    valid_opts = ['y', 'n']
    while save_opt not in valid_opts:
        save_opt = str(input("Should we save the ensemble? (y/n): ")).lower()
        if save_opt not in valid_opts:
            print("Sorry I didn't recognize that answer. (y/n)")
        else:
            save_opt = True if save_opt == 'y' else False
            break

    if save_opt:
        program_kwargs['save_name'] = input("Enter filename?: ")

    while load_opt not in valid_opts:
        load_opt = str(input("Should we attempt to load an ensemble? (y/n): ")).lower()
        if load_opt not in valid_opts:
            print("Sorry I didn't recognize that answer. (y/n)")
        else:
            load_opt = True if load_opt == 'y' else False
            break

    if load_opt:
        program_kwargs['load_name'] = input("Enter filename?: ")


    while type(program_kwargs['verbose']) != type(1):
        try:
            program_kwargs['verbose'] = input("How verbose should I be? (int, -1 (silent) - 2 (Everything)): ")
            program_kwargs['verbose'] = int(program_kwargs['verbose'])
        except:
            print("Sorry I didn't recognize that number.")
            program_kwargs['verbose'] = None

    # print()
    # print('Options')
    # print(program_kwargs)

    main(**program_kwargs)

if __name__ == "__main__":
    # main(dataset_name='finance', verbose=1, load_name='finance1', save_name='finance1')
    run_interactive()