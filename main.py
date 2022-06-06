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

    def read_csv_simple(self, string):
        with open(string, 'r') as file:
            csvreader = csv.reader(file)
            header = next(csvreader)
            rows = []
            for row in csvreader:
                rows.append(row)
        # result = [list(map(float, i)) for i in rows]
        result = np.array(rows)
        return result

    def zscore(self, data, mean, std):
        for i in range(len(data)):
            for j in range(len(data[0])):
                data[i][j] = (data[i][j] - mean[j]) / std[j]
        return data

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

    def y_hat(self, X, W):
        X = np.array(X)
        W = np.array(W)
        return np.dot(X, W)

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

        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        fmeasure = 2 * (recall * precision) / (recall + precision)
        accuracy = (TP + TN) / (TP + FP + FN + TN)
        return precision, recall, fmeasure, accuracy

    def norm_pdf(self, mu, sigma, x):
        scale = 1 / (sigma * math.sqrt(2 * math.pi))
        numerator = math.exp((-1 / 2) * (((x - mu) / sigma) ** 2))
        return scale * numerator

    def compute_probability(self, data, prior):
        sumlog = math.log(prior)
        # sumlog= prior[
        for i in range(len(data)):
            res = self.norm_pdf(0, 1, data[i])
            if res != 0:
                sumlog += math.log(res)
        return sumlog

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

    def convert_array_to_binary_values(self, data):
        data_mean = np.mean(data, axis=0)
        count = 0
        for i in range(len(data[0]) - 1):
            for j in range(len(data)):
                if data[j][i] >= data_mean[i]:
                    data[j][i] = count
                else:
                    data[j][i] = count + 1
            count += 2
        return data

    def convert_2d_int_to_string(self, data):
        result = []
        for i in range(len(data)):
            row = []
            for j in range(len(data[0])):
                row.append(str(data[i][j]))
            result.append(row)
        return np.array(result)

    def calc_total_entropy(self, train_data, label, class_list):
        total_row = train_data.shape[0] + 1
        total_entr = 0

        for c in class_list:
            total_class_count = train_data[train_data[label] == c].shape[0]
            total_class_entr = - (total_class_count / total_row) * np.log2(total_class_count / total_row)
            total_entr += total_class_entr

        return total_entr

    def calc_entropy(self, feature_value_data, label, class_list):
        class_count = feature_value_data.shape[0]
        entropy = 0

        for c in class_list:
            label_class_count = feature_value_data[feature_value_data[label] == c].shape[0]

            entropy_class = 0
            if label_class_count != 0:
                probability_class = label_class_count / class_count
                entropy_class = - probability_class * np.log2(probability_class)

            entropy += entropy_class

        return entropy

    def calc_info_gain(self, feature_name, train_data, label, class_list):
        feature_value_list = train_data[feature_name].unique()
        total_row = train_data.shape[0]
        feature_info = 0.0

        for feature_value in feature_value_list:
            feature_value_data = train_data[train_data[feature_name] == feature_value]
            feature_value_count = feature_value_data.shape[0]
            feature_value_entropy = self.calc_entropy(feature_value_data, label, class_list)
            feature_value_probability = feature_value_count / total_row
            feature_info += feature_value_probability * feature_value_entropy

        return self.calc_total_entropy(train_data, label, class_list) - feature_info

    def find_most_informative_feature(self, train_data, label, class_list):
        feature_list = train_data.columns.drop(label)
        max_info_gain = -1
        max_info_feature = None

        for feature in feature_list:
            feature_info_gain = self.calc_info_gain(feature, train_data, label, class_list)
            if max_info_gain < feature_info_gain:
                max_info_gain = feature_info_gain
                max_info_feature = feature

        return max_info_feature

    def generate_sub_tree(self, feature_name, train_data, label, class_list):
        feature_value_count_dict = train_data[feature_name].value_counts(sort=False)
        tree = {}

        for feature_value, count in feature_value_count_dict.iteritems():
            feature_value_data = train_data[train_data[feature_name] == feature_value]

            assigned_to_node = False
            for c in class_list:
                class_count = feature_value_data[feature_value_data[label] == c].shape[0]

                if class_count == count:
                    tree[feature_value] = c
                    train_data = train_data[train_data[feature_name] != feature_value]
                    assigned_to_node = True
            if not assigned_to_node:
                tree[feature_value] = "?"

        return tree, train_data

    def make_tree(self, root, prev_feature_value, train_data, label, class_list):
        if train_data.shape[0] != 0:
            max_info_feature = self.find_most_informative_feature(train_data, label, class_list)
            tree, train_data = self.generate_sub_tree(max_info_feature, train_data, label, class_list)
            next_root = None

            if prev_feature_value != None:
                root[prev_feature_value] = dict()
                root[prev_feature_value][max_info_feature] = tree
                next_root = root[prev_feature_value][max_info_feature]
            else:
                root[max_info_feature] = tree
                next_root = root[max_info_feature]

            for node, branch in list(next_root.items()):
                if branch == "?":
                    feature_value_data = train_data[train_data[max_info_feature] == node]
                    self.make_tree(next_root, node, feature_value_data, label, class_list)

    def id3(self, train_data_m, label):
        train_data = train_data_m.copy()
        tree = {}
        class_list = train_data[label].unique()
        self.make_tree(tree, None, train_data_m, label, class_list)

        return tree

    def predict(self, tree, instance):
        if not isinstance(tree, dict):
            return tree
        else:
            root_node = next(iter(tree))
            feature_value = instance[root_node]
            if feature_value in tree[root_node]:
                return self.predict(tree[root_node][feature_value], instance)
            else:
                return None

    def id3_precision_recall_fmeasure_accuracy(self, tree, test_data_m, label):
        TP = TN = FP = FN = 0
        for index, row in test_data_m.iterrows():
            result = self.predict(tree, test_data_m.iloc[index])
            if result == test_data_m[label].iloc[index] and float(result) == 1:
                TP += 1
            if result == test_data_m[label].iloc[index] and float(result) == 0:
                TN += 1
            if result != test_data_m[label].iloc[index] and float(result) == 1:
                FP += 1
            if result != test_data_m[label].iloc[index] and float(result) == 0:
                FN += 1

        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        fmeasure = 2 * (recall * precision) / (recall + precision)
        accuracy = (TP + TN) / (TP + FP + FN + TN)
        return precision, recall, fmeasure, accuracy

    def eval(self, tree, test_data_m, label):
        correct_preditct = 0
        wrong_preditct = 0
        for index, row in test_data_m.iterrows():
            result = self.predict(tree, test_data_m.iloc[index])
            if result == test_data_m[label].iloc[index]:
                correct_preditct += 1
            else:
                wrong_preditct += 1
        accuracy = correct_preditct / (correct_preditct + wrong_preditct)
        return accuracy

    def print_results(self, model, precision, recall, fmeasure, accuracy):
        print(model.capitalize() + " Algorithm Results: ")
        print("===========================================================")
        print("Precision, Recall and F-Measureand Accuracy of Validation Data: ", precision, recall, fmeasure, accuracy)
