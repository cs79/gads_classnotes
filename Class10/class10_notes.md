## Class 10 - Model Evaluation Metrics

### Model Evaluation Procedures (wrapup from Class 07)

Remember the process for building a model that generalizes well:

1) split dataset
2) train model on training data
3) test model on test data
4) tune parameters
5) choose best model
6) train on **all** data
7) make predictions on *new* data

#### Refresher: train_test_split and KNeighborsClassifier with iris data

```python
# revisiting iris knn training
from sklearn.datasets import load_iris
iris = load_iris()
X, y = iris.data, iris.target

from sklearn.neighbors import KNeighborsClassifier  # import estimator to use
knn = KNeighborsClassifier(n_neighbors=1)           # instantiate estimator w/ params
knn.fit(X, y)                                       # fit our model (train)
knn.score(X, y)                                     # test our model

from sklearn.cross_validation import train_test_split
# step 1: split data into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=4)
X_train.shape
X_test.shape
y_train.shape
y_test.shape

# steps 2 and 3: calculate test set error for K=1
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
knn.score(X_test, y_test)

# step 4 (parameter tuning): calculate test set error for K=5
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
knn.score(X_test, y_test)

# steps 5 and 6: choose best model (K=5) and train on all data
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X, y)

# step 7: make predictions on new ("out of sample") data
out_of_sample = [[5, 4, 3, 2], [4, 3, 2, 1]]
knn.predict(out_of_sample)

```
#### New material: Cross-Validation

Steps for K-fold cross-validation:

1) randomly split the dataset into K equal partitions
2) use partition 1 as test set, and the union of the other partitions as training set
3) calculate test set error
4) repeat steps 2-3 using a different partition as the test set in each iteration
5) take the average test set error as the estimate of out-of-sample accuracy

Features of K-fold cross-validation:

- more accurate estimate of out-of-sample prediction error
- more efficient use of data than a single train/test split
    - each record in our dataset is used for both training and testing
- presents tradeoff between efficiency and computational expense
    - 10-fold CV is 10x more expensive than a single train/test split
- can be used for parameter tuning and model selection

#### Cross-validation in scikit-learn

```python

# check CV score for K=1
from sklearn.cross_validation import cross_val_score
knn = KNeighborsClassifier(n_neighbors=1)
scores = cross_val_score(knn, X, y, cv=5, scoring='accuracy')
scores
np.mean(scores)

# check CV score for K=5
knn = KNeighborsClassifier(n_neighbors=5)
scores = cross_val_score(knn, X, y, cv=5, scoring='accuracy')
scores
np.mean(scores)

# search for an optimal value of K
k_range = range(1, 30, 2)
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores.append(np.mean(cross_val_score(knn, X, y, cv=5, scoring='accuracy')))
scores

# plot the K values (x-axis) versus the 5-fold CV score (y-axis)
plt.figure()
plt.plot(k_range, scores)

```

#### Using grid_search in sklearn

```python

# automatic grid search for an optimal value of K
from sklearn.grid_search import GridSearchCV
knn = KNeighborsClassifier()
k_range = range(1, 30, 2)
param_grid = dict(n_neighbors=k_range)
grid = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')
grid.fit(X, y)

# check the results of the grid search
grid.grid_scores_
grid_mean_scores = [result[1] for result in grid.grid_scores_]
plt.figure()
plt.plot(k_range, grid_mean_scores)
grid.best_score_
grid.best_params_
grid.best_estimator_

```
Grid search is a convenience method that runs a similar style loop as the above hand-coded example; we did not cover this in class but Kevin may record a video on it.  

### Model Evaluation Metrics

#### Regression: RMSE

- RMSE = sqrt(mean(squared errors))
-      = sqrt(1/n * sum_i_n(((yi - yi_hat)^2))

This is similar to minimization of squared error terms in linear regression, just more generalized.

#### RMSE in python

```python

## READ DATA AND CREATE DUMMY VARIABLES (this is just code from earlier class)

# read in the data
import pandas as pd
data = pd.read_csv('http://www-bcf.usc.edu/~gareth/ISL/Advertising.csv', index_col=0)

# create new feature 'Size', randomly assign as 'small' or 'large'
import numpy as np
np.random.seed(12345)
nums = np.random.rand(len(data))
mask_large = nums > 0.5
data['Size'] = 'small'
data.loc[mask_large, 'Size'] = 'large'

# create dummy variable 'IsLarge'
data['IsLarge'] = data.Size.map({'small':0, 'large':1})

# create new feature 'Area', randomly assign as 'rural' or 'suburban' or 'urban'
np.random.seed(123456)
nums = np.random.rand(len(data))
mask_suburban = (nums > 0.33) & (nums < 0.66)
mask_urban = nums > 0.66
data['Area'] = 'rural'
data.loc[mask_suburban, 'Area'] = 'suburban'
data.loc[mask_urban, 'Area'] = 'urban'

# create dummy variables 'Area_suburban' and 'Area_urban'
area_dummies = pd.get_dummies(data.Area, prefix='Area').iloc[:, 1:]
data = pd.concat([data, area_dummies], axis=1)

## CROSS-VALIDATION USING RMSE (this part is new)

# create X and y
feature_cols = ['TV', 'Radio', 'Newspaper', 'IsLarge', 'Area_suburban', 'Area_urban']
X = data[feature_cols]
y = data.Sales

# use 10-fold cross-validation to estimate RMSE when including all features
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import cross_val_score
lm = LinearRegression()
scores = cross_val_score(lm, X, y, cv=10, scoring='mean_squared_error')
np.mean(np.sqrt(-scores))

# repeat using only the 'meaningful' predictors, and watch our RMSE decrease
feature_cols = ['TV', 'Radio']
X = data[feature_cols]
scores = cross_val_score(lm, X, y, cv=10, scoring='mean_squared_error')
np.mean(np.sqrt(-scores))

```
Things you want to minimize: loss functions
Things you want to maximize: reward functions / gain functions

Scoring functions in sklearn are about maximization, hence scoring functions for things like RMSE are negative when they get spit out (you want to minimize RMSE, so instead it tries to maximize what would be -RMSE)

#### Classification: Confusion Matrices

First we read in the student credit card default data again and look at some rudimentary accuracy metrics:

```python
## READ DATA AND SPLIT INTO TRAIN/TEST (this is the student default dataset)

# read in the data
import pandas as pd
data = pd.read_csv('https://raw.githubusercontent.com/justmarkham/DAT3/master/data/Default.csv')

# create X and y
X = data[['balance']]
y = data.default

# split into train and test
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

## CALCULATE ACCURACY

# create logistic regression model
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

# predict and calculate accuracy in one step
logreg.score(X_test, y_test)

# predict in one step, calculate accuracy in a separate step
preds = logreg.predict(X_test)
from sklearn import metrics
print metrics.accuracy_score(y_test, preds)

# compare to null accuracy rate
y_test.mean()
1 - y_test.mean()

```

This isn't super useful on its own, because we have really skewed classes, and the null accuracy rate is really high.

A confusion matrix can give us better insight into "how good" our model is:

```python

## CONFUSION MATRIX

# print confusion matrix
print metrics.confusion_matrix(y_test, preds)

# nicer confusion matrix
from nltk import ConfusionMatrix
print ConfusionMatrix(list(y_test), list(preds))

# sensitivity: percent of correct predictions when reference value is 'default'
21 / float(58 + 21)
print metrics.recall_score(y_test, preds)   # metrics has other test scores it can run in a canned manner like this

# specificity: percent of correct predictions when reference value is 'not default'
2416 / float(2416 + 5)

# predict probabilities
import matplotlib.pyplot as plt
probs = logreg.predict_proba(X_test)[:, 1]  # THIS SEEMS LIKE A SUPER HANDY FUNCTION
plt.hist(probs)

# use 0.5 cutoff for predicting 'default'
import numpy as np
preds = np.where(probs > 0.5, 1, 0)
print ConfusionMatrix(list(y_test), list(preds))

# change cutoff for predicting default to 0.2
preds = np.where(probs > 0.2, 1, 0)
print ConfusionMatrix(list(y_test), list(preds))

# check accuracy, sensitivity, specificity
print metrics.accuracy_score(y_test, preds)
45 / float(34 + 45)
2340 / float(2340 + 81)

'''
What happened here?

When we shifted our threshold for predict default to 0.2 in our predictions:

- our accuracy went down (slightly)
- our specificity went down (slightly)
- our sensitivity went WAY up

versus when our cutoff was 0.5

'''

```

#### Sensitivity, Specificity, Precision

**Sensitivity (Recall)**: When the actual value is positive, how often is the prediction correct?

Formula: TP / (TP + FN)

**Specificity**: When the actual value is negative, how often is the prediction correct?

Formula: TN / (TN + FP)

**Precision** (Positive Predictive Value): Ratio of True Positives to all Predicted Positives

Formula: TP / (TP + FP)

**False Positive Rate**: 1 - Specificity

**F1 Score**:

Formula: (precision * recall / (precision + recall))

In medicine: Highly sensitive test first, followed by a highly specific test.  This helps understand the logic of the naming a bit.

#### Classification: ROC curves

There is an explanatory video from Kevin on this; re-read Provost & Fawcett section on this because it was really good.

Some ROC code:

```python

## ROC CURVES and AUC

# plot ROC curve
fpr, tpr, thresholds = metrics.roc_curve(y_test, probs)
plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')

# calculate AUC
print metrics.roc_auc_score(y_test, probs)

# use AUC as evaluation metric for cross-validation
from sklearn.cross_validation import cross_val_score
X = data[['balance']]
y = data.default
logreg = LogisticRegression()
cross_val_score(logreg, X, y, cv=10, scoring='roc_auc').mean()

# compare to a model with an additional feature
X = data[['balance', 'income']]
cross_val_score(logreg, X, y, cv=10, scoring='roc_auc').mean()

```

#### Notes to self:

logreg.predict_proba() seems like a very useful function
