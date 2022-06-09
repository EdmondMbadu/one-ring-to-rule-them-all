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
    gradient = sum([precision, recall, fmeasure, accuracy])
    model.print_results("Gradient Descent (Logistic Regression)", precision, recall, fmeasure, accuracy)
    print("Script running time in seconds: ", round(time.time() - start, 3))
    start = time.time()
    model = naive_bayes.NaiveBayes(X, Y, X_v, Y_v)
    precision, recall, fmeasure, accuracy = model.train()
    naive = sum([precision, recall, fmeasure, accuracy])
    model.print_results("Naive Bayes", precision, recall, fmeasure, accuracy)
    print("Script running time in seconds: ", round(time.time() - start, 3))
    start = time.time()
    model = decision_trees.DecisionTree(args.data)
    model.process()
    precision, recall, fmeasure, accuracy = model.train()
    tree = sum([precision, recall, fmeasure, accuracy])
    model.print_results("Decision Tree", precision, recall, fmeasure, accuracy)
    print("Script running time in seconds: ", round(time.time() - start, 3))

    if gradient >= naive and gradient >= tree:
        print("=============================================================================================")
        print("The Gradient descent algorithm is the most performant one with a summed value of ", round(gradient, 3),
              "naive with sum", round(naive, 3), " and tree with sum: ", round(tree, 3))
        print("=============================================================================================")
    elif naive > gradient and naive > tree:
        print("=============================================================================================")
        print("The Naive bayes algorithm is the most performant one with a summed value of ", round(naive, 3),
              "gradient with sum", round(gradient, 3), " and tree with sum: ", round(tree, 3))
        print("=============================================================================================")
    else:
        print("=============================================================================================")
        print("The decision tree (ID3) algorithm is the most performant one with a summed value of ", round(tree, 3),
              "gradient  with sum", round(gradient, 3), " and naive with sum: ", round(naive, 3))
        print("=============================================================================================")



else:
    print("Unknown model")
