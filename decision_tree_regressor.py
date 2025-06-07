import numpy as np
from asserts import asserts

class TreeNode():
    def __init__(self, left=None, right=None, feature_index=None, threshold=None, reduction_in_var=None, value=None):
        self.left = left
        self.right = right
        self.feature_index = feature_index
        self.threshold = threshold
        self.reduction_in_var = reduction_in_var
        self.value = value  #for leaf nodes

class DecisionTreeRegressor():
    def __init__(self, min_samples_split=2, max_depth=5):
        self.root = None
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth

    def get_best_split(self, X, y):
        n_samples, n_features = X.shape
        best_weighted_child_var = float('inf')
        best_feature_index = None
        best_threshold = None

        for i in range(n_features):
            sorted_col_indices = X[:, i].argsort()
            sorted_col = X[:, i][sorted_col_indices]
            sorted_y = y[sorted_col_indices]

            for j in range(n_samples - 1):
                threshold = (sorted_col[j] + sorted_col[j + 1]) / 2
                left_y = sorted_y[sorted_col <= threshold]
                right_y = sorted_y[sorted_col > threshold]

                if len(left_y) < 2 or len(right_y) < 2:
                    continue
                    
                weighted_left_var = np.var(left_y) * len(left_y) / n_samples
                weighted_right_var = np.var(right_y) * len(right_y) / n_samples
                weighted_child_var = weighted_left_var + weighted_right_var

                if weighted_child_var < best_weighted_child_var:
                    best_weighted_child_var = weighted_child_var
                    best_feature_index = i
                    best_threshold = threshold

        if best_feature_index is None:  #no valid split found
            return None, None, None
            
        reduction_in_var = np.var(y) - best_weighted_child_var
        return best_feature_index, best_threshold, reduction_in_var

    def build_tree(self, X, y, depth=0):
        n_samples = X.shape[0] if len(X.shape) > 1 else len(X)
        
        if n_samples < self.min_samples_split or depth >= self.max_depth:
            return TreeNode(value=np.mean(y))
        
        feature_index, threshold, reduction_in_var = self.get_best_split(X, y)
        if feature_index is None:  #no valid split found
            return TreeNode(value=np.mean(y))

        left_mask = X[:, feature_index] <= threshold
        right_mask = ~left_mask
        
        X_left, y_left = X[left_mask], y[left_mask]
        X_right, y_right = X[right_mask], y[right_mask]

        left_child = self.build_tree(X_left, y_left, depth + 1)
        right_child = self.build_tree(X_right, y_right, depth + 1)

        return TreeNode(left_child, right_child, feature_index, threshold, reduction_in_var)
    
    def fit(self, X_train, y_train):
        asserts(X_train, y_train)
            
        self.root = self.build_tree(X_train, y_train)

    def predict(self, X_test):
        if self.root is None:
            raise ValueError("The tree is not fitted yet")
            
        preds = np.zeros(len(X_test))
        for i in range(len(X_test)):
            node = self.root
            while node.value is None:
                if X_test[i, node.feature_index] <= node.threshold:
                    node = node.left
                else:
                    node = node.right
            preds[i] = node.value
        return preds

    def score(self, X_test, y_test):
        y_pred = self.predict(X_test)
        rss = np.sum((y_test - y_pred) ** 2)
        tss = np.sum((y_test - np.mean(y_test)) ** 2)
        return 1 - (rss / tss) if tss != 0 else 0.0