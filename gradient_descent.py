import numpy as np
import time
import main


class GradientDescent(main.Main):
    def __init__(self, X_train, Y_train, X_validation, Y_validation, lr, epochs):
        self.X = X_train
        self.Y = Y_train
        self.X_v = X_validation
        self.Y_v = Y_validation
        self.lr = lr
        self.limit = epochs

    def convert_bias_to_column_in_matrix(self, matrix):
        transformed = []
        matrix = np.array(matrix)
        if matrix.ndim > 1:
            for i in range(len(matrix)):
                row = []
                row.append(1)
                for j in range(len(matrix[i])):
                    row.append(matrix[i][j])
                transformed.append(row)
        else:
            ones = np.ones(len(matrix))
            transformed = np.array(transformed)
            transformed = np.vstack((ones, matrix)).T

        return np.array(transformed)

    def train_model(self, X, W, Y, n, epoch, limit, learning_rate):
        Y = Y.reshape(len(Y), 1)
        X = self.convert_bias_to_column_in_matrix(X)
        while epoch < limit:
            Y_hat = self.y_hat(X, W)
            Y_hat = 1 / (1 + np.exp(-Y_hat))
            A = np.transpose(X)
            B = (Y_hat - Y)
            djdW = (1 / n) * np.dot(A, B)
            W = W - (learning_rate * djdW)
            epoch += 1
        return W

    def train(self):
        Y_hat = np.array([])
        N = len(self.X[0]) + 1
        J = np.array([])
        n = len(self.Y)
        learning_rate = self.lr
        W = np.random.uniform(low=-0.0001, high=0.0001, size=(N, 1))
        limit = self.limit
        model = self.train_model(self.X, W, self.Y, n, 0, limit, learning_rate)
        X = self.convert_bias_to_column_in_matrix(self.X)
        Y_hat = self.y_hat(X, model)
        Y_hat = 1 / (1 + np.exp(-Y_hat))
        # This is in case we need the training results
        # precision, recall, fmeasure, accuracy = main.precision_recall_fmeasure_accuracy(Y_hat, Y_train, .5)
        X = self.convert_bias_to_column_in_matrix(self.X_v)
        Y_hat_v = self.y_hat(X, model)
        Y_hat_v = 1 / (1 + np.exp(-Y_hat_v))
        return self.precision_recall_fmeasure_accuracy(Y_hat_v, self.Y_v, .4)

    def y_hat(self, X, W):
        X = np.array(X)
        W = np.array(W)
        return np.dot(X, W)
