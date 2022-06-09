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
        multiplier = 1
        if len(self.data) > 1000:
            multiplier = .01
        if len(self.data) < 1000 and len(self.data) > 200:
            multiplier = .05
        if len(self.data) < 200:
            multiplier = .025
        if len(self.data) < 100:
            multiplier = .2
        ratio = len(self.data) * multiplier
        self.training = self.data[:round((len(self.data)) / ratio), :]
        self.validation = self.data[round((2 * len(self.data)) / 3):, :]

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

        precision = TP / (TP + FP) if (TP + FP) else TN
        recall = TP / (TP + FN) if (TP + FN) else TN
        fmeasure = 2 * (recall * precision) / (recall + precision) if (recall + precision) else (recall + precision)
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
