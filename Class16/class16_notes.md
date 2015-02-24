## Class 16 - Ensembling

### PANDAS NOTE:

Axis 1: concat as additional columns
Axis 0: concat as additional rows

Good to know.

### Decision Trees wrap-up

#### Titanic data - exercise in Python:

```python

# read in the data
titanic = pd.read_csv('https://raw.githubusercontent.com/justmarkham/DAT4/master/data/titanic.csv')

# look for missing values
titanic.isnull().sum()

# encode sex feature
titanic['sex'] = titanic.sex.map({'female':0, 'male':1}) ## this seems handy (.map())

# fill in missing values for age
titanic.age.fillna(titanic.age.mean(), inplace=True)

# create three dummy variables, drop the first dummy variable, and store this as a DataFrame (dropping first is convention; only need 2)
embarked_dummies = pd.get_dummies(titanic.embarked, prefix='embarked').iloc[:, 1:]

# concatenate the two dummy variable columns onto the original DataFrame
# note: axis=0 means rows, axis=1 means columns
titanic = pd.concat([titanic, embarked_dummies], axis=1)

# create a list of feature columns
feature_cols = ['pclass', 'sex', 'age', 'embarked_Q', 'embarked_S']

# define X and y
X = titanic[feature_cols]
y = titanic.survived

# fit a classification tree with max_depth=3 on all data
from sklearn.tree import DecisionTreeClassifier
treeclf = DecisionTreeClassifier(max_depth=3, random_state=1)
treeclf.fit(X, y)

# compute the feature importances
pd.DataFrame({'feature':feature_cols, 'importance':treeclf.feature_importances_})

```

#### Gini Coefficient

Measure of node purity; lower coefficient = purer node.  Classification trees are splitting to minimize the Gini Coefficient on potential resultant nodes.

NB: splits may result in two nodes with the overall same predicted outcome, which seems "pointless" in some sense, but it was split to increase node purity.  It is also perfectly possible that a subsequent split on those branches would make the predicted outcomes no longer the same, even though the two parent nodes would have predicted the same outcome -- their four children might not all agree.

#### Decision trees - advantages and disadvantages

**Advantages:**
- Can be specified as a series of rules, and are thought to more closely approximate human decision-making than other models
- Non-parametric (will do better than linear regression if relationship between predictors and response is highly non-linear)

**Disadvantages:**
- Small variations in the data can result in a completely different tree
- Recursive binary splitting makes "locally optimal" decisions that may not result in a globally optimal tree
- Can create biased trees if the classes are highly imbalanced

### Ensembling

Building (independent) models, and aggregating their predictions (perhaps through something as simple as a majority vote).

As demonstrated in the example in class using random data, this improves predictive accuracy even if all voting models have the same base accuracy.

From iPython Notebook:

**Ensemble learning (or "ensembling")** is simply the process of combining several models to solve a prediction problem, with the goal of producing a combined model that is more accurate than any individual model. For **classification** problems, the combination is often done by majority vote. For **regression** problems, the combination is often done by taking an average of the predictions.

For ensembling to work well, the individual models must meet two conditions:

- Models should be **accurate** (they must outperform random guessing)
- Models should be **independent** (their predictions are not correlated with one another)

The idea, then, is that if you have a collection of individually imperfect (and independent) models, the "one-off" mistakes made by each model are probably not going to be made by the rest of the models, and thus the mistakes will be discarded when averaging the models.

It turns out that as you add more models to the voting process, the probability of error decreases. This is known as Condorcet's Jury Theorem, which was developed by a French political scientist in the 18th century.

#### Bootstrapping

The basic idea of bootstrapping is that inference about a population from sample data (sample → population) can be modeled by resampling the sample data and performing inference on (resample → sample). As the population is unknown, the true error in a sample statistic against its population value is unknowable. In bootstrap-resamples, the 'population' is in fact the sample, and this is known; hence the quality of inference from resample data → 'true' sample is measurable.

More formally, the bootstrap works by treating inference of the true probability distribution J, given the original data, as being analogous to inference of the empirical distribution of Ĵ, given the resampled data. The accuracy of inferences regarding Ĵ using the resampled data can be assessed because we know Ĵ. If Ĵ is a reasonable approximation to J, then the quality of inference on J can in turn be inferred.

**Example:**

```python
# set a seed for reproducibility
np.random.seed(1)

# create an array of 0 to 9, then sample 10 times with replacement
np.random.choice(a=10, size=10, replace=True)
```

#### Bagging

General-purpose procedure that can be applied to many models (including decision trees to create random forests).

**Process:**
- Take repeated bootstrap samples (random samples with replacement) from the training data set
- Train our method on each boostrapped training set and make predictions
- Average the predictions

This increases predictive accuracy by **reducing the variance**, similar to how cross-validation reduces the variance associated with the test set approach (for estimating out-of-sample error) by splitting many times an averaging the results.

#### Applying bagging to decision trees

So how exactly can bagging be used with decision trees? Here's how it applies to regression trees:

- Grow B regression trees using B bootstrapped training sets
- Grow each tree deep so that each one has low bias
- Every tree makes a numeric prediction, and the predictions are averaged (to reduce the variance)

It is applied in a similar fashion to classification trees, except that during the prediction stage, the overall prediction is based upon a majority vote of the trees.

What value should be used for B? Simply use a large enough value that the error seems to have stabilized. (Choosing a value of B that is "too large" will generally not lead to overfitting.)

#### Manually implementing bagged decision trees (with B=3)

```python

import pandas as pd

# read in vehicle data
vehicles = pd.read_csv('https://raw.githubusercontent.com/justmarkham/DAT4/master/data/used_vehicles.csv')

# convert car to 0 and truck to 1
vehicles['type'] = vehicles.type.map({'car':0, 'truck':1})

# calculate the number of rows in vehicles
n_rows = vehicles.shape[0]

# set a seed for reproducibility
np.random.seed(123)

# create three bootstrap samples (will be used to select rows from the DataFrame)
sample1 = np.random.choice(a=n_rows, size=n_rows, replace=True)
sample2 = np.random.choice(a=n_rows, size=n_rows, replace=True)
sample3 = np.random.choice(a=n_rows, size=n_rows, replace=True)

# use sample1 to select rows from DataFrame
print vehicles.iloc[sample1, :]

from sklearn.tree import DecisionTreeRegressor

# grow one regression tree with each bootstrapped training set
treereg1 = DecisionTreeRegressor(random_state=123)
treereg1.fit(vehicles.iloc[sample1, 1:], vehicles.iloc[sample1, 0])

treereg2 = DecisionTreeRegressor(random_state=123)
treereg2.fit(vehicles.iloc[sample2, 1:], vehicles.iloc[sample2, 0])

treereg3 = DecisionTreeRegressor(random_state=123)
treereg3.fit(vehicles.iloc[sample3, 1:], vehicles.iloc[sample3, 0])

# read in out-of-sample data
oos = pd.read_csv('https://raw.githubusercontent.com/justmarkham/DAT4/master/data/used_vehicles_oos.csv')

# convert car to 0 and truck to 1
oos['type'] = oos.type.map({'car':0, 'truck':1})

# select feature columns (every column except for the 0th column)
feature_cols = vehicles.columns[1:]

# make predictions on out-of-sample data
preds1 = treereg1.predict(oos[feature_cols])
preds2 = treereg2.predict(oos[feature_cols])
preds3 = treereg3.predict(oos[feature_cols])

# print predictions
print preds1
print preds2
print preds3

# average predictions and compare to actual values
print (preds1 + preds2 + preds3)/3
print oos.price.values

```

#### Estimating out-of-sample error

Bagged models have a very nice property: **out-of-sample error can be estimated without using the test set approach or cross-validation!**

Here's how the out-of-sample estimation process works with bagged trees:

- On average, each bagged tree uses about two-thirds of the observations. **For each tree, the remaining observations are called "out-of-bag" observations.**
- For the first observation in the training data, predict its response using **only the trees in which that observation was out-of-bag.** Average those predictions (for regression) or take a majority vote (for classification).
- Repeat this process for every observation in the training data.
- Compare all predictions to the actual responses in order to compute a mean squared error or classification error. This is known as the **out-of-bag error.**

**When B is sufficiently large, the out-of-bag error is an accurate estimate of out-of-sample error.**

```python

# set is a data structure used to identify unique elements
print set(range(14))

# only show the unique elements in sample1
print set(sample1)

# use the "set difference" to identify the out-of-bag observations for each tree
print sorted(set(range(14)) - set(sample1))
print sorted(set(range(14)) - set(sample2))
print sorted(set(range(14)) - set(sample3))

```

Thus, we would predict the response for **observation 4** by using tree 1 (because it is only out-of-bag for tree 1). We would predict the response for **observation 5** by averaging the predictions from trees 1, 2, and 3 (since it is out-of-bag for all three trees). We would repeat this process for all observations, and then calculate the MSE using those predictions.

#### Estimating variable importance

Although bagging **increases predictive accuracy**, it **decreases model interpretability** because it's no longer possible to visualize the tree to understand the importance of each variable.

However, we can still obtain an overall summary of "variable importance" from bagged models:

- To compute variable importance for bagged regression trees, we can calculate the **total amount that the mean squared error is decreased due to splits over a given predictor, averaged over all trees.**
- A similar process is used for bagged classification trees, except we use the Gini index instead of the mean squared error.

### Random Forests

Random Forests is a slight variation of bagged trees that has even better performance! Here's how it works:

- Exactly like bagging, we create an ensemble of decision trees using bootstrapped samples of the training set.
- However, when building each tree, each time a split is considered, a random sample of m predictors is chosen as split candidates from the full set of p predictors. The split is only allowed to use one of those m predictors.

Notes:

- A new random sample of predictors is chosen for every single tree at every single split.
- For classification, m is typically chosen to be the square root of p. For regression, m is typically chosen to be somewhere between p/3 and p.

What's the point?

- Suppose there is one very strong predictor in the data set. When using bagged trees, most of the trees will use that predictor as the top split, resulting in an ensemble of similar trees that are "highly correlated".
- Averaging highly correlated quantities does not significantly reduce variance (which is the entire goal of bagging).
- By randomly leaving out candidate predictors from each split, Random Forests "decorrelates" the trees, such that the averaging process can reduce the variance of the resulting model.

#### IDEA:

Another potential benefit: since individual trees can't "look ahead" due to greedy selection, this forces them not to make the same greedy choices in parallel, giving potentially "better" trees with somewhat different upper nodes the chance to grow and have their votes counted in the ensemble.

#### Random Forests in Python

```python

# read in the Titanic data
titanic = pd.read_csv('https://raw.githubusercontent.com/justmarkham/DAT4/master/data/titanic.csv')

# encode sex feature
titanic['sex'] = titanic.sex.map({'female':0, 'male':1})

# fill in missing values for age
titanic.age.fillna(titanic.age.mean(), inplace=True)

# create three dummy variables, drop the first dummy variable, and store this as a DataFrame
embarked_dummies = pd.get_dummies(titanic.embarked, prefix='embarked').iloc[:, 1:]

# concatenate the two dummy variable columns onto the original DataFrame
# note: axis=0 means rows, axis=1 means columns
titanic = pd.concat([titanic, embarked_dummies], axis=1)

# create a list of feature columns
feature_cols = ['pclass', 'sex', 'age', 'embarked_Q', 'embarked_S']

# print the updated DataFrame
titanic.head(10)

# import class, instantiate estimator, fit with all data
from sklearn.ensemble import RandomForestClassifier
rfclf = RandomForestClassifier(n_estimators=100, max_features='auto', oob_score=True, random_state=1)
rfclf.fit(titanic[feature_cols], titanic.survived)

# compute the feature importances
pd.DataFrame({'feature':feature_cols, 'importance':rfclf.feature_importances_})

# compute the out-of-bag classification accuracy
rfclf.oob_score_

```

#### Most important tuning parameters for Random Forests:

- **n_estimators:** more estimators (trees) increases performance but decreases speed
- **max_features:** cross-validate to choose an ideal value
