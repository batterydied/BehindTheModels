import numpy as np

class LinearRegression():
    def __init__(self, learning_rate= 0.01, max_iteration= 1000, tolerance= 1e-6):
        self.weights = None
        self.bias = 0
        self.learning_rate = learning_rate
        self.max_iteration = max_iteration
        self.tolerance = tolerance

    def fit(self, X_train, y_train):
        n_samples, n_features = X_train.shape
        self.weights = np.random.randn(n_features)

        for _ in range(self.max_iteration):
            preds = X_train @ self.weights + self.bias
            error = preds - y_train

            mse = np.mean(error ** 2)
            print(f'MSE: {mse}')

            dW = (2/n_samples) * X_train.T @ error
            dB = (2/n_samples) * np.sum(error)
            
            update_w = self.learning_rate * dW
            update_b = self.learning_rate * dB

            if np.all(np.abs(update_w) < self.tolerance) and (abs(update_b) < self.tolerance):
                break

            self.weights -= update_w
            self.bias -= update_b

        return [self.weights, self.bias]
    
    def predict(self, X_test):
        return X_test @ self.weights + self.bias
    
    def score(self, X_test, y_test):
        rss = np.sum((y_test - self.predict(X_test)) ** 2)
        tss = np.sum((y_test - np.mean(y_test)) ** 2)
        return 1 - rss/tss

