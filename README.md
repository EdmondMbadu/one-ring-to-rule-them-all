# one-ring-to-rule-them-all

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
