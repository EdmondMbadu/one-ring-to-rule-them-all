import numpy as np
import csv
import math
import argparse


class Main:

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

    def zscore(self, data, mean, std):
        for i in range(len(data)):
            for j in range(len(data[0])):
                data[i][j] = (data[i][j] - mean[j]) / std[j]
        return data

    def precision_recall_fmeasure_accuracy(self, Y_hat, Y, treshold):
        TP = TN = FP = FN = 0
        guess = -1
        for i in range(len(Y)):
            if Y_hat[i] >= treshold:
                guess = 1
            else:
                guess = 0
            if guess == Y[i][0] and guess == 1:
                TP += 1
            if guess == Y[i][0] and guess == 0:
                TN += 1
            if guess != Y[i][0] and guess == 1:
                FP += 1
            if guess != Y[i][0] and guess == 0:
                FN += 1

        precision = TP / (TP + FP) if (TP + FP) else TP
        recall = TP / (TP + FN) if (TP + FN) else TP
        fmeasure = 2 * (recall * precision) / (recall + precision) if (recall + precision) else (recall + precision)
        accuracy = (TP + TN) / (TP + FP + FN + TN)
        return precision, recall, fmeasure, accuracy

    def print_results(self, model, precision, recall, fmeasure, accuracy):
        print(model.capitalize() + " Algorithm Results: ")
        print("===========================================================")
        print("Precision, Recall and F-Measureand Accuracy of Validation Data: ", precision, recall, fmeasure, accuracy)
