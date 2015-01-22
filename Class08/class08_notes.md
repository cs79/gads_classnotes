## Class 08 - Linear Regression

Using iPython notebook here:
http://nbviewer.ipython.org/github/justmarkham/DAT4/blob/master/notebooks/08_linear_regression.ipynb

```python
# Advertising data example

data = pd.read_csv('http://www-bcf.usc.edu/~gareth/ISL/Advertising.csv', index_col=0)

# visualize the relationship between the features and the response using scatterplots
# revisit Python for Data Analysis (visualization chapter) for how to use these axes
fig, axs = plt.subplots(1, 3, sharey=True)
data.plot(kind='scatter', x='TV', y='Sales', ax=axs[0], figsize=(16, 8))
data.plot(kind='scatter', x='Radio', y='Sales', ax=axs[1])
data.plot(kind='scatter', x='Newspaper', y='Sales', ax=axs[2])

# use statsmodels to estimate model coefficients:
import statsmodels.formula.api as smf
lm = smf.ols(formula='Sales ~ TV', data=data).fit()

# print the coefficients:
lm.params

# from this we can see that a one-unit increase in TV ad spending is associated with a 0.047537 unit increase in sales

# use statsmodels to predict:
# you have to create a DataFrame since the Statsmodels formula interface expects it
X_new = pd.DataFrame({'TV': [50, 95, 12, 430], 'Radio': [12, 3, 12, 4], 'Newspaper': [1, 2, 3, 4]})
X_new.head()

# use the model to make predictions on a new value
lm.predict(X_new)   # returns an array with predicted values

# create a DataFrame with the minimum and maximum values of TV
X_new = pd.DataFrame({'TV': [data.TV.min(), data.TV.max()]})
X_new.head()
# make predictions for those x values and store them
preds = lm.predict(X_new)
preds

# first, plot the observed data
data.plot(kind='scatter', x='TV', y='Sales')

# then, plot the least squares line
plt.plot(X_new, preds, c='red', linewidth=2)

```

Confidence in our Model
Question: Is linear regression a high bias/low variance model, or a low variance/high bias model?

Answer: High bias/low variance. Under repeated sampling, the line will stay roughly in the same place (low variance), but the average of those models won't do a great job capturing the true relationship (high bias). **Note that low variance is a useful characteristic when you don't have a lot of training data!**

A closely related concept is confidence intervals. Statsmodels calculates 95% confidence intervals for our model coefficients, which are interpreted as follows: If the population from which this sample was drawn was sampled 100 times, approximately 95 of those confidence intervals would contain the "true" coefficient.

```python

# print the p-values for the model coefficients
lm.pvalues

# this tells us the probability that we would have observed the coefficients we did, if the data had no true relationship

```

The most common way to evaluate the overall fit of a linear model is by the R-squared value. **R-squared is the proportion of variance explained, meaning the proportion of variance in the observed data that is explained by the model, or the reduction in error over the null model.** (The null model just predicts the mean of the observed response, and thus it has an intercept and no slope.)

R-squared is between 0 and 1, and higher is better because it means that more variance is explained by the model.

```python
# print the R-squared value for the model
lm.rsquared
```
Is that a "good" R-squared value? It's hard to say. The threshold for a good R-squared value depends widely on the domain. **Therefore, it's most useful as a tool for comparing different models.**

### Multiple (Multivariate) Linear Regression

```python

# create a fitted model with all three features
lm = smf.ols(formula='Sales ~ TV + Radio + Newspaper', data=data).fit()

# print the coefficients
lm.params

# print a summary
lm.summary()

```

#### Interpreting coefficients of multiple linear regression

Without interactions:
- each slope indicates, **holding the other features fixed**, the relationship between that feature and the target


### Feature Selection

How do I decide which features to include in a linear model? Here's one idea:

- Try different models, and only keep predictors in the model if they have small p-values.
- Check whether the R-squared value goes up when you add new predictors.

What are the drawbacks to this approach?

- Linear models rely upon a lot of assumptions (such as the features being independent), and if those assumptions are violated (which they usually are), R-squared and p-values are less reliable.
- Using a p-value cutoff of 0.05 means that if you add 100 predictors to a model that are pure noise, 5 of them (on average) will still be counted as significant.
- R-squared is susceptible to overfitting, and thus there is no guarantee that a model with a high R-squared value will generalize.

**R-squared will always increase as you add more features to the model, even if they are unrelated to the response.** Thus, selecting the model with the highest R-squared is not a reliable approach for choosing the best linear model.

### Linear Regression in scikit-learn

```python

# create X and y
feature_cols = ['TV', 'Radio', 'Newspaper']
X = data[feature_cols]      #if you make this a list, you can zip it with coefs later...
y = data.Sales

# follow the usual sklearn pattern: import, instantiate, fit
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X, y)

# print intercept and coefficients
print lm.intercept_
print lm.coef_

# pair the feature names with the coefficients
zip(feature_cols, lm.coef_) #...like so

# predict for a new observation
lm.predict([100, 25, 25])

# calculate the R-squared
lm.score(X, y)

```

### Handling Categorical Predictors with Two Categories

Up to now, all of our predictors have been numeric. What if one of our predictors was categorical?

Let's create a new feature called Size, and randomly assign observations to be small or large:

```python

import numpy as np

# set a seed for reproducibility
np.random.seed(12345)

# create a Series of booleans in which roughly half are True
nums = np.random.rand(len(data))
mask_large = nums > 0.5

# initially set Size to small, then change roughly half to be large
data['Size'] = 'small'
data.loc[mask_large, 'Size'] = 'large'
data.head()

# create a new Series called IsLarge (this is a dummy var)
data['IsLarge'] = data.Size.map({'small':0, 'large':1})
data.head()

# Let's redo the multiple linear regression and include the IsLarge predictor:
# create X and y
feature_cols = ['TV', 'Radio', 'Newspaper', 'IsLarge']
X = data[feature_cols]
y = data.Sales

# instantiate, fit
lm = LinearRegression()
lm.fit(X, y)

# print coefficients
zip(feature_cols, lm.coef_)

```
#### Coefficient interpretation:

How do we interpret the IsLarge coefficient? For a given amount of TV/Radio/Newspaper ad spending, being a large market is associated with an average increase in Sales of 57.42 widgets (as compared to a Small market, which is called the baseline level).

What if we had reversed the 0/1 coding and created the feature 'IsSmall' instead? The coefficient would be the same, except it would be negative instead of positive. As such, your choice of category for the baseline does not matter, all that changes is your interpretation of the coefficient.


### Handling Categorical Predictors with more than two categories

```python

# set a seed for reproducibility
np.random.seed(123456)

# assign roughly one third of observations to each group
nums = np.random.rand(len(data))
mask_suburban = (nums > 0.33) & (nums < 0.66)
mask_urban = nums > 0.66
data['Area'] = 'rural'
data.loc[mask_suburban, 'Area'] = 'suburban'
data.loc[mask_urban, 'Area'] = 'urban'
data.head()

```

We have to represent Area numerically, but we can't simply code it as 0=rural, 1=suburban, 2=urban because that would imply an ordered relationship between suburban and urban (and thus urban is somehow "twice" the suburban category).

Instead, we create another dummy variable:

```python

# create three dummy variables using get_dummies, then exclude the first dummy column
area_dummies = pd.get_dummies(data.Area, prefix='Area').iloc[:, 1:] #positional indexing with R-like syntax [r, c] where : means all r / c

# concatenate the dummy variable columns onto the original DataFrame (axis=0 means rows, axis=1 means columns)
data = pd.concat([data, area_dummies], axis=1)
data.head()

```

Here is how we interpret the coding:

- rural is coded as Area_suburban=0 and Area_urban=0
- suburban is coded as Area_suburban=1 and Area_urban=0
- urban is coded as Area_suburban=0 and Area_urban=1

Why do we only need two dummy variables, not three? Because two dummies captures all of the information about the Area feature, and implicitly defines rural as the baseline level. (In general, if you have a categorical feature with k levels, you create k-1 dummy variables.)

If this is confusing, think about why we only needed one dummy variable for Size (IsLarge), not two dummy variables (IsSmall and IsLarge).

Let's include the two new dummy variables in the model:

```python

# create X and y
feature_cols = ['TV', 'Radio', 'Newspaper', 'IsLarge', 'Area_suburban', 'Area_urban']
X = data[feature_cols]
y = data.Sales

# instantiate, fit
lm = LinearRegression()
lm.fit(X, y)

# print coefficients
zip(feature_cols, lm.coef_)

```

#### Interpretation of coefficients

How do we interpret the coefficients?

- Holding all other variables fixed, being a suburban area is associated with an average decrease in Sales of 106.56 widgets (as compared to the baseline level, which is rural).

- Being an urban area is associated with an average increase in Sales of 268.13 widgets (as compared to rural).

A final note about dummy encoding: If you have categories that can be ranked (i.e., strongly disagree, disagree, neutral, agree, strongly agree), you can potentially use a single dummy variable and represent the categories numerically (such as 1, 2, 3, 4, 5).
