import main
import numpy as np
import pandas as pd
import sys, threading
import time
import pickle

start = time.time()


class DecisionTree(main.Main):

    def __init__(self, csv):
        self.data_name = csv
        self.data = np.array([])
        self.training = []
        self.validation = []
        self.validation = []

    def process(self):
        self.data = self.read_csv(self.data_name)
        np.random.seed(0)
        np.random.shuffle(self.data)
        self.training = self.data[:round((len(self.data)) / 50), :]
        self.validation = self.data[round((2 * len(self.data)) / 3):, :]

    def train(self):
        self.training = self.convert_array_to_binary_values(self.training)
        self.validation = self.convert_array_to_binary_values(self.validation)
        row = np.arange(len(self.training[0]))
        self.training = self.convert_2d_int_to_string(self.training)
        self.validation = self.convert_2d_int_to_string(self.validation)
        result = pd.DataFrame(self.training, columns=row)
        validation = pd.DataFrame(self.validation, columns=row)
        tree = self.id3(result, result.shape[1] - 1)
        # learned_model = tree
        return self.id3_precision_recall_fmeasure_accuracy(tree, validation,
                                                           result.shape[1] - 1)

# data = main.read_csv("breast_cancer.csv")
# # 2. Seeds the random generator with zero and shuffles the observations
#
# # 3. Select the first 2/3 round up of the data for training and the remaining for validation
# training = data[:round((len(data)) / 30), :]
# validation = data[round((2 * len(data)) / 3):, :]
# print(training.shape)
# validation = data[len(training):, :]
# 4. zscore features using training data
# training = main.convert_array_to_binary_values(training)
# validation = main.convert_array_to_binary_values(validation)
# row = np.arange(len(training[0]))
#
# training = main.convert_2d_int_to_string(training)
# validation = main.convert_2d_int_to_string(validation)
# result = pd.DataFrame(training, columns=row)
# validation = pd.DataFrame(validation, columns=row)
#
# tree = main.id3(result, result.shape[1] - 1)
#
# learned_model = tree

# with open('learned_model.data', 'wb') as f:
# retrieve the model
# learned_model = pickle.load(f)
# save the model ( when running for the first time
# pickle.dump(learned_model, f)

# precision, recall, fmeasure, accuracy = main.id3_precision_recall_fmeasure_accuracy(tree, validation,
#                                                                                     result.shape[1] - 1)
# print("Precision, Recall and F-Measureand Accuracy of Testing Data: ", precision, recall, fmeasure, accuracy)
# print("Script Running time in seconds: ", round(time.time() - start, 3))
