import numpy as np
import pandas as pd
import csv


class DataProcessor:
    def __init__(self, name):
        self.data = self.read_csv(name)

    def process(self):
        np.random.seed(0)
        np.random.shuffle(self.data)
        X_train = self.data[:round((2 / 3) * len(self.data)), :len(self.data[0]) - 1]
        Y_train = self.data[:round((2 / 3) * len(self.data)), len(self.data[0]) - 1:len(self.data[0])]
        X_validation = self.data[len(X_train):, :len(self.data[0]) - 1]
        Y_validation = self.data[len(X_train):, len(self.data[0]) - 1:len(self.data[0])]
        X_train_mean = np.mean(X_train, axis=0)
        X_train_std = np.std(X_train, axis=0)
        X_train_Zscored = self.zscore(X_train.copy(), X_train_mean, X_train_std)
        X_validation_Zscored = self.zscore(X_validation.copy(), X_train_mean, X_train_std)
        return X_train_Zscored, Y_train, X_validation_Zscored, Y_validation

    def zscore(self, data, mean, std):
        for i in range(len(data)):
            for j in range(len(data[0])):
                data[i][j] = (data[i][j] - mean[j]) / std[j]
        return data

    def read_csv(self, string):
        with open(string, 'r') as file:
            csvreader = csv.reader(file)
            header = next(csvreader)
            rows = []
            for row in csvreader:
                rows.append(row)
        result = [list(map(float, i)) for i in rows]
        result = np.array(result)
        return result
