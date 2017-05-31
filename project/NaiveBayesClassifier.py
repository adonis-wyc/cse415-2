import math
import numpy as np

# def separate_for_boosting(X, Y, preds):
#     correct = np.array([(x,y) for x, y, p in zip(X, Y, preds) if y == p])
#     incorrect = np.array([(x,y) for x, y, p in zip(X, Y, preds) if y != p])
#     correct, incorrect = correct.T, incorrect.T
#     return correct, incorrect

def separateByClass(dataset):
    # takes the last col as Y and groups the dataframe by that col
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if (vector[-1] not in separated):
            separated[vector[-1]] = []
        separated[vector[-1]].append(vector)
    return separated

def summarize(dataset):

    summaries = [(np.mean(attribute), np.std(attribute)) for attribute in zip(*dataset)]
    del summaries[-1]
    return summaries

def summarizeByClass(dataset):
    # separates the dataset into each class and then stores the summary (mean, std) of each variable in the dict for every class in the dataset
    separated = separateByClass(dataset)
    summaries = {}
    for classValue, instances in separated.items():
        summaries[classValue] = summarize(instances)
    return summaries

def calculateProbability(x, mean, stdev):
#     print(stdev)
    if stdev == 0: stdev = 0.0000001
    exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
    return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent


class GenericNB():
    def __init__(self):
        self.summaries = dict

    def fit(self, X, Y):
        Y = Y.reshape((1,-1)) if len(Y.shape) == 1 else Y.T
        dataset = np.concatenate([X.T, Y]).T
        self.summaries = summarizeByClass(dataset)

    def predict(self, X):
        predictions = []
        for i in range(len(X)):
            result = self._predict(X[i])
            predictions.append(result)
        return predictions

    def predict_proba(self, X):
        return np.array([list(self.calculateClassProbabilities(x).values()) for x in X])

    def score(self, X, Y):
        predictions = self.predict(X)
        correct = 0
        for i in range(len(Y)):
            if Y[i] == predictions[i]:
                correct += 1
        return (correct/float(len(Y)))

    def _predict(self, X):
        probabilities = self.calculateClassProbabilities(X)
        bestLabel, bestProb = None, -1
        for classValue, probability in probabilities.items():
            if bestLabel is None or probability > bestProb:
                bestProb = probability
                bestLabel = classValue
        return bestLabel

    def calculateClassProbabilities(self, X):
        probabilities = {}
        for classValue, classSummaries in self.summaries.items():
            probabilities[classValue] = 1
            for i in range(len(classSummaries)):
                mean, stdev = classSummaries[i]
                x = X[i]
                probabilities[classValue] *= calculateProbability(x, mean, stdev)
        return probabilities
