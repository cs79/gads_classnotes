## Class 07 - scikit-learn, Model Evaluation Procedures

### scikit-learn for analysis of iris dataset

```python
# imports and setup
from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

iris = load_iris() # special sklearn function to conveniently load the iris dataset

X, y = iris.data, iris.target
X.shape
y.shape

""" predict y with KNN
The below are "generalized steps" for using basically ANY model in sklearn:
1: import the class you need AND ONLY THE CLASS(ES) YOU NEED -- DO NOT IMPORT THE WHOLE THING
2: instantiate the estimator [WITH PARAMETERS] -- the estimator instance won't "do" anything on its own, it's kind of like an re pattern
3: fit the data with features, response (almost all estimators in sklearn have a .fit method) -- no need to assign to anything else, fit method just updates the instance (I think?)
4: predict for a new observation
"""
from sklearn.neighbors import KNeighborsClassifier  # import class
knn = KNeighborsClassifier(n_neighbors=1)           # instantiate the estimator
knn.fit(X, y)                                       # fit with data
knn.predict([3, 5, 4, 2])                           # predict for a new observation
iris.target_names[knn.predict([3, 5, 4, 2])]
knn.predict([3, 5, 2, 2])

# predict for multiple observations at once
X_new = [[3, 5, 4, 2], [3, 5, 2, 2]]
knn.predict(X_new)

# try a different value of K
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X, y)
knn.predict(X_new)              # predictions
knn.predict_proba(X_new)        # predicted probabilities - useful method (probably even moreso for other sklearn estimators)
knn.kneighbors([3, 5, 4, 2])    # distances to nearest neighbors (and identities)
np.sqrt(((X[106] - [3, 5, 4, 2])**2).sum()) # Euclidian distance calculation for nearest neighbor

# compute the accuracy for K=5 and K=1
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X, y)
knn.score(X, y)                             # accuracy of predicting its own training data wrt the model's own y values, used to train them
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X, y)
knn.score(X, y)

```

### Model Evaluation

```python

## TRAIN AND TEST ON THE SAME DATA (OVERFITTING)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X, y)
knn.score(X, y)


## TEST SET APPROACH

# understanding train_test_split
from sklearn.cross_validation import train_test_split
features = np.array([range(10), range(10, 20)]).T
response = ['even', 'odd'] * 5
features_train, features_test = train_test_split(features, random_state=5) # random state just seeds the rng within the function call
features_train
features_test
features_train, features_test, response_train, response_test = train_test_split(features, response, random_state=1)
features_train
features_test
response_train
response_test

# step 1: split data into training set and test set
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=4)

# steps 2 and 3: calculate test set error for K=1
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)                   # train on training
knn.score(X_test, y_test)                   # score on test

# step 4 (parameter tuning): calculate test set error for K=5
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
knn.score(X_test, y_test)

# try a bunch of shit:
def train_on_k(k, X_train, y_train, X_test, y_test):
    results = Series(index = np.arange(k)+1)
    for i in range(k):
        knn = KNeighborsClassifier(n_neighbors=i+1)
        knn.fit(X_train, y_train)
        results.iloc[i] = knn.score(X_test, y_test)
    return results

train_on_k(k=50, X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test)

# steps 5 and 6: choose best model (K=5) and train on all data
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(iris.data, iris.target)

# step 7: make predictions on new ("out of sample") data
out_of_sample = [[5, 4, 3, 2], [4, 3, 2, 1]]
knn.predict(out_of_sample)

# verify that a different train/test split can result in a different test set error
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=1)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
knn.score(X_test, y_test)


```
