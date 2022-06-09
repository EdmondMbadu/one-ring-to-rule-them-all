# one-ring-to-rule-them-all for binary classification


##About the Program:

We attempt to automate the process of finding which machine learning algorithm performs best given a learning task. To start, we have implemented the gradient descent (with logistic regression), decision tree (ID3), and naıve bayes (Gaussian). We planned to extend the implementation to inlucde more algorithm and multi class classification in the near future. 





## How to run the program on your local machine: 
To run all the algorithms simultaneously for a given dataset run the following script:

```
python3 execute.py all <name of the dataset>
```

For instance to run the breast cancer dataset do the following:

```
python3 execute.py all breast_cancer.csv
```

To run a particular algorithm run the following script:

```
python3 execute.py <name of the algorithm> <name of the dataset> 
```

For instance to run the diabetes dataset do the following:

```
python3 execute.py gradient diabetes.csv --lr .001 --epochs 10000 
```

The name of the algorithms in the context of the script are : gradient, naive, and tree. The options of adding –-lr and
–-epochs is only possible for the gradient descent algorithm as it is the only algorithm that has the flexibility of
adjusting hyperparameters.

__Note that you can add a csv dataset and get the predictions given that all the features are real ( or integer)values
and that the column to predict is the last and that it is binary.__
We have 5 available datasets in this directory that are readily available to be used: 
breast_cancer.csv, caesarian.csv, diabetes.csv, divorce.csv, spambase.data. 

## Sample Results:
<img width="1223" alt="Screen Shot 2022-06-08 at 9 45 09 PM" src="https://user-images.githubusercontent.com/36565046/172746536-c14e74c9-f667-4d6f-8f89-694b1862f1ac.png">




