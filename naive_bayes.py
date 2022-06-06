import numpy as np
import time
import math

import main


class NaiveBayes(main.Main):

    def __init__(self, X_train, Y_train, X_validation, Y_validation):
        self.X = X_train
        self.Y = Y_train
        self.X_v = X_validation
        self.Y_v = Y_validation

    def compute_probability(self, data, prior):
        sumlog = math.log(prior)
        # sumlog= prior[
        for i in range(len(data)):
            res = self.norm_pdf(0, 1, data[i])
            if res != 0:
                sumlog += math.log(res)
        return sumlog

    def norm_pdf(self, mu, sigma, x):
        scale = 1 / (sigma * math.sqrt(2 * math.pi))
        numerator = math.exp((-1 / 2) * (((x - mu) / sigma) ** 2))
        return scale * numerator

    def divide_into_two_groups(self, X, Y):
        positive_x = []
        positive_y = []
        negative_x = []
        negative_y = []
        for i in range(len(Y)):
            if Y[i] == 1:
                positive_x.append(X[i])
                positive_y.append(Y[i])
            else:
                negative_x.append(X[i])
                negative_y.append(Y[i])
        return np.array(positive_x), np.array(positive_y), np.array(negative_x), np.array(negative_y)

    def train(self):
        x_positive, y_positive, x_negative, y_negative = self.divide_into_two_groups(self.X, self.Y)
        prior_positive = len(x_positive) / len(self.X)
        prior_negative = len(x_negative) / len(self.X)
        positive_mean = np.mean(x_positive, axis=0)
        negative_mean = np.mean(x_negative, axis=0)
        Y_hat = self.y_hat_naive_bayes(self.X_v, prior_positive, prior_negative, x_positive, x_negative)
        return self.precision_recall_fmeasure_accuracy(Y_hat, self.Y_v, 0.5)
        # self.print_results("Naive Bayes", precision, recall, fmeasure, accuracy)

    def divide_into_two_groups(self, X, Y):
        positive_x = []
        positive_y = []
        negative_x = []
        negative_y = []
        for i in range(len(Y)):
            if Y[i] == 1:
                positive_x.append(X[i])
                positive_y.append(Y[i])
            else:
                negative_x.append(X[i])
                negative_y.append(Y[i])
        return np.array(positive_x), np.array(positive_y), np.array(negative_x), np.array(negative_y)

    def y_hat_naive_bayes(self, X, prior_positive, prior_negative, x_positive, x_negative):
        Y_hat = []
        positive_mean = np.mean(x_positive, axis=0)
        negative_mean = np.mean(x_negative, axis=0)
        positive_std = np.std(x_positive, axis=0)
        negative_std = np.std(x_negative, axis=0)
        X_positive = self.zscore(X.copy(), positive_mean, positive_std)
        X_negative = self.zscore(X.copy(), negative_mean, negative_std)
        for i in range(len(X)):
            prob_positive = self.compute_probability(X_positive[i], prior_positive)
            prob_negative = self.compute_probability(X_negative[i], prior_negative)
            if prob_positive >= prob_negative:
                Y_hat.append(1)
            else:
                Y_hat.append(0)
        return np.array(Y_hat)
