import numpy as np

class DecisionTreeClassifierCustom():
    def __init__(self, max_depth=10, min_count=10, is_binary=False):
        self.tree = None
        self.max_depth = max_depth
        self.min_count = min_count
        self.train_x = None
        self.train_y = None
        self.is_binary = is_binary

    def train(self, X, y):
        self.train_x = [(x, i) for i, x in enumerate(X)]
        self.train_y = y
        self.n_features = X.shape[1]
        root = self._get_branch(self.train_x, self.train_y)
        self._branch(root, self.train_y, 1)
        self.tree = root

    def fit(self, X, y):
        return self.train(X, y)

    def score(self, X, Y):
        predictions = self.predict(X)
        correct = 0
        for i in range(len(Y)):
            if Y[i] == predictions[i]:
                correct += 1
        return (correct/float(len(Y)))

    def predict(self, X_test):
        pred_y = np.empty(X_test.shape[0])
        for i, obs in enumerate(X_test):
            pred_y[i] = self._predict_helper(self.tree, obs)
        return pred_y

    def _predict_helper(self, node, obs):
        if obs[node['index']] <= node['value']:
            if isinstance(node['left'], dict):
                return self._predict_helper(node['left'], obs)
            else:
                return node['left']
        else:
            if isinstance(node['right'], dict):
                return self._predict_helper(node['right'], obs)
            else:
                return node['right']

    def _get_branch(self, X, y):
        classes = list(np.unique(y))
#         print('classes', classes)
        b_index, b_value, b_score, b_groups = np.inf, np.inf, np.inf, None
        for index in range(self.n_features):
            for row, idx in X:
                groups = self._test_split(index, 0.5 if self.is_binary else row[index], X)
                gini = self._get_gini(groups, classes, y)
                if gini < b_score:
#                     print(gini)
                    b_index, b_value, b_score, b_groups = index, 0.5 if self.is_binary else row[index], gini, groups
#                     print(b_index, b_value, b_score)
        return {'index':b_index, 'value':b_value, 'groups':b_groups}

    def _test_split(self, index, value, X):
        less = []
        more = []
        for row, idx in X:
            if row[index] <= value:
                less.append((row, idx))
            else:
                more.append((row, idx))
        return less, more

    def _get_gini(self, groups, classes, y):
        gini = 0.0
        for cls in classes:
            for group in groups:
                size = len(group)
                if size == 0:
#                     gini += 0.025
                    continue
                proportion = [y[x[1]] for x in group].count(cls) / float(size)
                gini += (proportion * (1.0 - proportion))
        return gini

    def _branch(self, node, y, depth):
        left, right = node['groups']
        del(node['groups'])
        # check for a no split
        if not left or not right:
            node['left'] = self._to_leaf(left + right, y)
            node['right'] = self._to_leaf(left + right, y)

            return
        # check for max depth
        if depth >= self.max_depth:
            node['left'], node['right'] = self._to_leaf(left, y), self._to_leaf(right, y)
            return
        # process left child
        if len(left) <= self.min_count:
            node['left'] = self._to_leaf(left, y)
        else:
            node['left'] = self._get_branch(left, y)
            self._branch(node['left'], y, depth+1)
        # process right child
        if len(right) <= self.min_count:
            node['right'] = self._to_leaf(right, y)
        else:
            node['right'] = self._get_branch(right, y)
            self._branch(node['right'], y, depth+1)

    def _to_leaf(self, group, y):
        outcomes = [y[x[1]] for x in group]
        return max(set(outcomes), key=outcomes.count)