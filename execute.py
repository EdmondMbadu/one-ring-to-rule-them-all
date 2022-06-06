import argparse
import time

import decision_trees
import gradient_descent
import data_processing
import naive_bayes
import main

parser = argparse.ArgumentParser(
    description='Trains and evaluates the given model.')
parser.add_argument('model', type=str, help='Model to use for training')
parser.add_argument('data', type=str, default='breast_cancer.csv', help='The name of the csv data')
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
parser.add_argument('--epochs', type=int, default=10000, help='Number of training epochs')
args = parser.parse_args()
if args.model == "gradient":
    start = time.time()
    data = data_processing.DataProcessor(args.data)
    X, Y, X_v, Y_v = data.process()
    model = gradient_descent.GradientDescent(X, Y, X_v, Y_v, args.lr, args.epochs)
    precision, recall, fmeasure, accuracy = model.train()
    model.print_results("Gradient Descent (Logistic Regression)", precision, recall, fmeasure, accuracy)
    print("Script running time in seconds: ", round(time.time() - start, 3))

elif args.model == "naive":
    start = time.time()
    data = data_processing.DataProcessor(args.data)
    X, Y, X_v, Y_v = data.process()
    model = naive_bayes.NaiveBayes(X, Y, X_v, Y_v)
    precision, recall, fmeasure, accuracy = model.train()
    model.print_results("Naive Bayes", precision, recall, fmeasure, accuracy)
    print("Script running time in seconds: ", round(time.time() - start, 3))

elif args.model == "tree":
    start = time.time()
    model = decision_trees.DecisionTree(args.data)
    model.process()
    precision, recall, fmeasure, accuracy = model.train()
    model.print_results("Deciion Tree", precision, recall, fmeasure, accuracy)
    print("Script running time in seconds: ", round(time.time() - start, 3))
elif args.model == 'all':
    start = time.time()
    data = data_processing.DataProcessor(args.data)
    X, Y, X_v, Y_v = data.process()
    model = gradient_descent.GradientDescent(X, Y, X_v, Y_v, args.lr, args.epochs)
    precision, recall, fmeasure, accuracy = model.train()
    model.print_results("Gradient Descent (Logistic Regression)", precision, recall, fmeasure, accuracy)
    print("Script running time in seconds: ", round(time.time() - start, 3))
    start = time.time()
    model = naive_bayes.NaiveBayes(X, Y, X_v, Y_v)
    precision, recall, fmeasure, accuracy = model.train()
    model.print_results("Naive Bayes", precision, recall, fmeasure, accuracy)
    print("Script running time in seconds: ", round(time.time() - start, 3))
    start = time.time()
    model = decision_trees.DecisionTree(args.data)
    model.process()
    precision, recall, fmeasure, accuracy = model.train()
    model.print_results("Deciion Tree", precision, recall, fmeasure, accuracy)
    print("Script running time in seconds: ", round(time.time() - start, 3))


else:
    print("Unknown model")
