import numpy as np

class LinearRegression():
    def __init__(self, learning_rate = 0.01, max_epochs = 100, treshold = 1e-6, batch_size = None):
        self.weights = None
        self.bias = 0
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.treshold = treshold
        self.batch_size = batch_size

    def fit(self, X_train, y_train):
        n_samples, n_features = X_train.shape
        self.weights = np.random.randn(n_features)

        batch_size = self.batch_size if self.batch_size else n_samples
        previous_loss = float('-inf')

        for _ in range(self.max_epochs):
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            X_train = X_train[indices]
            y_train = y_train[indices]
            total_loss = 0

            for i in range(0, n_samples, batch_size):
                samples = X_train[i: i + batch_size]
                true_vals = y_train[i: i + batch_size]

                dW, dB = self.gradient_descent(samples, true_vals)
                
                update_w = self.learning_rate * dW
                update_b = self.learning_rate * dB

                self.weights -= update_w
                self.bias -= update_b

                error = samples @ self.weights + self.bias - true_vals
                total_loss += np.sum(error ** 2)
            
            epoch_loss = total_loss / n_samples
            if abs(epoch_loss - previous_loss) < self.treshold:
                break
            previous_loss = epoch_loss
            
        return [self.weights, self.bias]
    
    def gradient_descent(self, samples, true_vals):
        n_samples = samples.shape[0]
        preds = samples @ self.weights + self.bias
        error = preds - true_vals
        #mse = np.mean(error ** 2)
        #print(f'MSE: {mse}')

        dW = (2/n_samples) * samples.T @ error
        dB = (2/n_samples) * np.sum(error)

        return [dW, dB]

    def predict(self, X_test):
        return X_test @ self.weights + self.bias
    
    def score(self, X_test, y_test):
        rss = np.sum((y_test - self.predict(X_test)) ** 2)
        tss = np.sum((y_test - np.mean(y_test)) ** 2)
        return 1 - rss/tss

