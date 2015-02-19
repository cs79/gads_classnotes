## Class 15 - CART, Ensembling

### Building a regression tree using recursive binary splitting

1. Begin at top of tree
2. For every single predictor, examine every possible cutpoint, and choose the predictor / cutpoint combination that has the lowest MSE
    - MSE is measured across each terminus versus the training points that match that terminus
3. Repeat (1, 2) at the resulting two nodes per branch
4. Repeat recursively until stopping condition is met

### Determining when to stop

- Can use a stopping criterion such a "maximum depth" or "minimum samples per leaf"
- Can also grow a deliberately deeper-than-desired tree, then "prune" it

### Classification Trees in scikit-learn:

```python

import pandas as pd

# read in vehicle data
vehicles = pd.read_csv('https://raw.githubusercontent.com/justmarkham/DAT4/master/data/used_vehicles.csv')

# convert car to 0 and truck to 1
vehicles['type'] = vehicles.type.map({'car':0, 'truck':1})

# select feature columns (every column except for the 0th column)
feature_cols = vehicles.columns[1:]

# define X (features) and y (response)
X = vehicles[feature_cols]
y = vehicles.price

# split into train/test
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# print out each of the arrays
print X_train
print y_train
print X_test
print y_test

# import class, instantiate estimator, fit with training set
from sklearn.tree import DecisionTreeRegressor
treereg = DecisionTreeRegressor(random_state=1)
treereg.fit(X_train, y_train)

# make predictions
preds = treereg.predict(X_test)

# print predictions and actual values
print preds
print y_test

# print RMSE
from sklearn import metrics
import numpy as np
np.sqrt(metrics.mean_squared_error(y_test, preds))

# use cross-validation to find best max_depth
from sklearn.cross_validation import cross_val_score

# try max_depth=2
treereg = DecisionTreeRegressor(max_depth=2, random_state=1)
scores = cross_val_score(treereg, X, y, cv=3, scoring='mean_squared_error')
np.mean(np.sqrt(-scores))

# try max_depth=3
treereg = DecisionTreeRegressor(max_depth=3, random_state=1)
scores = cross_val_score(treereg, X, y, cv=3, scoring='mean_squared_error')
np.mean(np.sqrt(-scores))

# try max_depth=4
treereg = DecisionTreeRegressor(max_depth=4, random_state=1)
scores = cross_val_score(treereg, X, y, cv=3, scoring='mean_squared_error')
np.mean(np.sqrt(-scores))

# max_depth=3 was best, so fit a tree using that parameter with ALL DATA
treereg = DecisionTreeRegressor(max_depth=3, random_state=1)
treereg.fit(X, y)

# compute the "Gini importance" of each feature: the (normalized) total reduction of MSE brought by that feature
pd.DataFrame({'feature':feature_cols, 'importance':treereg.feature_importances_})

# read in out-of-sample data
oos = pd.read_csv('https://raw.githubusercontent.com/justmarkham/DAT4/master/data/used_vehicles_oos.csv')

# convert car to 0 and truck to 1
oos['type'] = oos.type.map({'car':0, 'truck':1})

# define X and y
X_oos = oos[feature_cols]
y_oos = oos.price

# make predictions on out-of-sample data
preds = treereg.predict(X_oos)

# print predictions and actual values
print preds
print y_oos.values

# print RMSE
np.sqrt(metrics.mean_squared_error(y_oos, preds))

# print RMSE for the tree you created!
your_preds = [2050, 5000, 18000]
np.sqrt(metrics.mean_squared_error(y_oos, your_preds))

other_preds = [2722, 2722, 13500]
np.sqrt(metrics.mean_squared_error(y_oos, other_preds))

```
