## Class 11: Working on a Data Problem
## Alex, Roger, Ganesh

### Approach:
- clean if necessary
- make categorical features (for each of 5 forward return periods)
- train categorical models
    - feature selection
    - normalization of certain features
- train regression models
    - feature selection
    - normalization of certain features

### Vars of notable interest:
- ZYX1MinSentiment:     raw SUM of sentiment on Twitter in last minute (re: ZYX tweets)
- ZYX1MinTweets:        number of tweets in last min
- similar for other intervals (rolling sum of sentiment, number of tweets in that period)
- ZYX1minPriceChange:   pct change in price in the last minute


## Guiding questions:

1. Regression vs. classification - how?

2. Accuracy vs. sensitivity vs. specificity

3. Long model vs. short model

4. Create new predictors (from existing data) -- what sorts of things?
    - average sentiment per window
    -
    - maybe take Price out of the features
    - do anything like bin the times (morning/afternoon/evening trade?)

5. What other predictors could improve this model?

### Work:

```python
## imports:
from pandas import DataFrame, Series
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib qt
from scipy.stats import zscore
from sklearn.cross_validation import train_test_split, cross_val_score


url = 'https://raw.githubusercontent.com/justmarkham/DAT4/master/data/ZYX_prices.csv'
raw = pd.read_csv(url, header=0)

X = raw.iloc[:,np.arange(1,20)]
y = raw.iloc[:,np.arange(20,25)]

## create some new features:

X['1MinAvgSentiment'] = X.ZYX1MinSentiment / X.ZYX1MinTweets
X['5MinAvgSentiment'] = X.ZYX5minSentiment / X.ZYX5minTweets
X['10MinAvgSentiment'] = X.ZYX10minSentiment / X.ZYX10minTweets
X['20MinAvgSentiment'] = X.ZYX20minSentiment / X.ZYX20minTweets
X['30MinAvgSentiment'] = X.ZYX30minSentiment / X.ZYX30minTweets
X['60MinAvgSentiment'] = X.ZYX60minSentiment / X.ZYX60minTweets

## create a normed version (because I think it'll predict better)
X_normed = X.apply(zscore, axis=0)

## Train / Test Split for EACH y
X_train, X_test, y0_train, y0_test = train_test_split(X, y[0], random_state=123)
X_train, X_test, y1_train, y1_test = train_test_split(X, y[1], random_state=123)
X_train, X_test, y2_train, y2_test = train_test_split(X, y[2], random_state=123)
X_train, X_test, y3_train, y3_test = train_test_split(X, y[3], random_state=123)
X_train, X_test, y4_train, y4_test = train_test_split(X, y[4], random_state=123)

# ## build some regression models
#
# from sklearn.linear_model import LinearRegression
# linreg0 = LinearRegression()
# linreg0.fit(X, y['5fret'])
# linreg0.score(X, y['5fret'])
#
# linreg1 = LinearRegression()
# linreg1.fit(X, y['10fret'])
# linreg1.score(X, y['10fret'])
#
# linreg2 = LinearRegression()
# linreg2.fit(X, y['20fret'])
# linreg2.score(X, y['20fret'])
#
# linreg3 = LinearRegression()
# linreg3.fit(X, y['30fret'])
# linreg3.score(X, y['30fret'])
#
# linreg4 = LinearRegression()
# linreg4.fit(X, y['60fret'])
# linreg4.score(X, y['60fret'])

## build a classification model:

y_bin_1 = Series([1 if y['5fret'][x] > 0 else 0 for x in range(len(y))])
y_bin_2 = Series([1 if y['10fret'][x] > 0 else 0 for x in range(len(y))])
y_bin_3 = Series([1 if y['20fret'][x] > 0 else 0 for x in range(len(y))])
y_bin_4 = Series([1 if y['30fret'][x] > 0 else 0 for x in range(len(y))])
y_bin_5 = Series([1 if y['60fret'][x] > 0 else 0 for x in range(len(y))])

y_binary = pd.concat([y_bin_1, y_bin_2, y_bin_3, y_bin_4, y_bin_5], axis=1)

## try Logistic Regression
X_train, X_test, y0_train, y0_test = train_test_split(X, y[0], random_state=123)
X_train, X_test, y1_train, y1_test = train_test_split(X, y[1], random_state=123)
X_train, X_test, y2_train, y2_test = train_test_split(X, y[2], random_state=123)
X_train, X_test, y3_train, y3_test = train_test_split(X, y[3], random_state=123)
X_train, X_test, y4_train, y4_test = train_test_split(X, y[4], random_state=123)


from sklearn.linear_model import LogisticRegression

#5 min
mod0 = LogisticRegression()
scores0 = cross_val_score(mod0, X, y_binary.iloc[:,0], cv=5, scoring='accuracy')

mod0_normed = LogisticRegression()
scores0_normed = cross_val_score(mod0_normed, X_normed, y_binary.iloc[:,0], cv=5, scoring='accuracy')

# 10 min
mod1 = LogisticRegression()
scores1 = cross_val_score(mod1, X, y_binary.iloc[:,1], cv=5, scoring='accuracy')

mod1_normed = LogisticRegression()
scores1_normed = cross_val_score(mod1_normed, X_normed, y_binary.iloc[:,1], cv=5, scoring='accuracy')

# 20 min
mod2 = LogisticRegression()
scores2 = cross_val_score(mod2, X, y_binary.iloc[:,2], cv=5, scoring='accuracy')

mod2_normed = LogisticRegression()
scores2_normed = cross_val_score(mod2_normed, X_normed, y_binary.iloc[:,2], cv=5, scoring='accuracy')

# 30 min
mod3 = LogisticRegression()
scores3 = cross_val_score(mod3, X, y_binary.iloc[:,3], cv=5, scoring='accuracy')

mod3_normed = LogisticRegression()
scores3_normed = cross_val_score(mod3_normed, X_normed, y_binary.iloc[:,3], cv=5, scoring='accuracy')

# 60 min
mod4 = LogisticRegression()
scores4 = cross_val_score(mod4, X, y_binary.iloc[:,4], cv=5, scoring='accuracy')

mod4_normed = LogisticRegression()
scores4_normed = cross_val_score(mod4_normed, X_normed, y_binary.iloc[:,4], cv=5, scoring='accuracy')

print([np.mean(scores0), np.mean(scores1), np.mean(scores2), np.mean(scores3), np.mean(scores4)])


print([np.mean(scores0_normed), np.mean(scores1_normed), np.mean(scores2_normed), np.mean(scores3_normed), np.mean(scores4_normed)])

# try some alternative features
X_copy = X.fillna(0)
X_copy_normed = X_copy.apply(zscore, axis=0)

mod_avg = LogisticRegression()
scores_avg = cross_val_score(mod_avg, X_copy, y_binary.iloc[:,4], cv=5, scoring='accuracy')
print np.mean(scores_avg)

```

### things to do to improve performance:

- Get feature combinations working properly
- use OLS to look for individually significant features
    - potentially try a few interactions if they seem to make sense
- other random derivative features
- correlation matrices
- small scatter matrices
- get more outside data to include in the model
- use other classification models
- try classifying into more than just a binary variable
