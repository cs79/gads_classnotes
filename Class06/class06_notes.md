## Class 06 Notes: numpy and KNN
### 1/12/2015

### numpy

ndarray is a staple numpy data structure; can be n-dimensional up to (some max that you will never get to because your brain will hurt)

pandas data structures like DataFrame, Series are built on top of ndarrays

```python
data1 = [6, 7.5, 8, 0, 1]           # list
arr1 = np.array(data1)              # 1d array
data2 = [range(1, 5), range(5, 9)]  # list of lists
arr2 = np.array(data2)              # 2d array
arr2.tolist()                       # ndarray method to convert array back to list

# examining arrays using ndarray methods (and len())
arr1.dtype      # float64
arr2.dtype      # int32
arr2.ndim       # 2
arr2.shape      # (2, 4) - axis 0 is rows, axis 1 is columns
arr2.size       # 8 - total number of elements
len(arr2)       # 2 - size of first dimension (aka axis)

# create special arrays
np.zeros(10)
np.zeros((3, 6))                # MUST pass a tuple in parens, unlike .reshape()
np.ones(10)
np.linspace(0, 1, 5)            # 0 to 1 (inclusive) with 5 points
np.logspace(0, 3, 4)            # 10^0 to 10^3 (inclusive) with 4 points -- This seems like it could be very interesting

# arange is like range, except it returns an array (not a list)
int_array = np.arange(5)
float_array = int_array.astype(float)   # also seems pretty useful

# slicing
arr1[0]         # 0th element (slices like a list)
arr2[0]         # row 0: returns 1d array ([1, 2, 3, 4])
arr2[0, 3]      # row 0, column 3: returns 4
arr2[0][3]      # alternative syntax
arr2[:, 0]      # all rows, column 0: returns 1d array ([1, 5])
arr2[:, 0:1]    # all rows, column 0: returns 2d array ([[1], [5]])

# views and copies
arr = np.arange(10)
arr[5:8]                    # returns [5, 6, 7]
arr[5:8] = 12               # all three values are overwritten (would give error on a list)
arr_view = arr[5:8]         # creates a "view" on arr, not a copy
arr_view[:] = 13            # modifies arr_view AND arr
arr_copy = arr[5:8].copy()  # makes a copy instead
arr_copy[:] = 14            # only modifies arr_copy

# using boolean arrays
names = np.array(['Bob', 'Joe', 'Will', 'Bob'])
names == 'Bob'                          # returns a boolean array
names[names != 'Bob']                   # logical selection
(names == 'Bob') | (names == 'Will')    # keywords "and/or" don't work with boolean arrays
names[names != 'Bob'] = 'Joe'           # assign based on a logical selection
np.unique(names)                        # set function

# vectorized operations
nums = np.arange(5)
nums*10                             # multiply each element by 10
nums = np.sqrt(nums)                # square root of each element
np.ceil(nums)                       # also floor, rint (round to nearest int)
np.isnan(nums)                      # checks for NaN
nums + np.arange(5)                 # add element-wise
np.maximum(nums, np.array([1, -2, 3, -4, 5]))  # compare element-wise

# math and stats
rnd = np.random.randn(4, 2) # random normals in 4x2 array
rnd.mean()
rnd.std()
rnd.argmin()                # index of minimum element
rnd.sum()
rnd.sum(axis=0)             # sum of columns
rnd.sum(axis=1)             # sum of rows

# use numpy to create scatter plots
N = 50

x = np.random.rand(N)
y = np.random.rand(N)
colors = np.random.rand(N)
area =30+(170*np.random.rand(N)) # 30 to 100 point radiuses - can scale randoms generated with np.random.rand() by multiplying / adding like this, for example

plt.scatter(x, y, s=area, c=colors,)
plt.show()

"""
Skipped in class:

# conditional logic
np.where(rnd > 0, 2, -2)    # args: condition, value if True, value if False
np.where(rnd > 0, 2, rnd)   # any of the 3 arguments can be an array

# methods for boolean arrays
(rnd > 0).sum()             # counts number of positive values
(rnd > 0).any()             # checks if any value is True
(rnd > 0).all()             # checks if all values are True

"""
# reshape, transpose, flatten
nums = np.arange(32).reshape(8, 4) # creates 8x4 array
nums.T                       # transpose
nums.flatten()               # flatten

# random numbers
np.random.seed(12234)
np.random.rand(2, 3)      # 0 to 1, in the given shape
np.random.randn(10)         # random normals (mean 0, sd 1)
np.random.randint(0, 2, 10) # 0 or 1

```

### KNN

Using the iris dataset to learn about KNN learning algorithm; original code in DAT4 file

```python

from sklearn.datasets import load_iris
from pandas import DataFrame, Series
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# load the famous iris data
iris = load_iris()

# what do you think these attributes represent?
iris.data                   # the actual dataset
iris.data.shape             # dimensions of the dataset
iris.feature_names          # names of the features of the dataset (as list) - basically the column headers, separated from the data matrix
iris.target                 # classification of the species
iris.target_names           # names of the species' as a list

# intro to numpy
type(iris.data)

## PART 1: Read data into pandas and explore


# the feature_names are a bit messy, let's clean them up. remove the (cm)
# at the end and replace any spaces with an underscore
# create a list "features" that holds the cleaned column names

features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

# a hard way to do this programatically:
features = [name[:-5].replace(' ', '_') for name in iris.feature_names]

# read the iris data into pandas, with our refined column names

df = DataFrame(iris.data, columns = features)

# create a list of species (should be 150 elements)
# using iris.target and iris.target_names
# resulting list should only have the words "setosa", "versicolor", and "virginica"

species = [iris.target_names[x] for x in iris.target]

# add the species list as a new DataFrame column

df['species'] = species

# explore data numerically, looking for differences between species
# try grouping by species and check out the different predictors

df.groupby('species').describe()

df.groupby('species').agg(np.mean)

# explore data by sorting, looking for differences between species

df.sort_index(by='sepal_length').values
df.sort_index(by='petal_width').values #interesting

# I used values in order to see all of the data at once
# without .values, a dataframe is returned


# explore data visually, looking for differences between species
# try using a histogram or boxplot

df.petal_length.hist(by=df.species, sharex=True)
df.petal_width.hist(by=df.species, sharex=True)

df.boxplot(by='species') #this one is mad useful for seeing "non-overlaps"

#how about this one:
data.groupby('species').mean().plot(kind='bar')

pd.scatter_matrix(df, c = iris.target)

## PART 2: Write a function to predict the species for each observation

# create a dictionary so we can reference columns by name

col_ix = {col:index for index, col in enumerate(df.columns)}

# define function that takes in a row of data and returns a predicted species

def classify_iris(data):
    if data[col_ix['petal_length']] < 3:
        return 'setosa'
    elif data[col_ix['petal_width']] < 1.8:
        return 'versicolor'
    else:
        return 'virginica'

# make predictions and store as numpy array

preds = np.array([classify_iris(row) for row in df.values])

# calculate the accuracy of the predictions

np.mean(preds == df.species.values) # mean works here ONLY because the values are binary

```

### Machine Learning

Class of algos that are data-driven; allow computers to "learn" without being explicitly programmed

#### Supervised learning
- Main purpose: making predictions
- requires target labels to train

Typical elements of supervised learning:
- Matrix of predictors X
- response y (target)
    - if y is continuous: Regression problem
    - if y is categorical: Classification problem
- data is composed of "Observations" (rows): a vector of predictors and the associated response

Supervised learning uses training cases to predict test cases

*we want to understand which predictors ACTUALLY effect the response, and how*

#### Unsupervised learning
- Main purpose: extracting structure
- does not require target labels to train anything

Typical elements of an unsupervised learning problem:
- no response variable y, just a set of predictors X
- objective is more open:
    - find groups of observations that behave similarly
    - find predictors that behave similarly
    - find combinations of features that explain the variation in the data
- difficult to evaluate how well you are doing
- data is easier to obtain (usually), since it doesn't need to be labeled
- sometimes useful as a preprocessing step for supervised learning (like PCA cutting out some highly correlated features before having a training algo attack some unnecessarily large dataset)
- common techniques: clustering, PCA
- if data are continuous: dimension reduction problem (usually)
- if data are categorical: clustering problem (usually)

#### KNN algo

1) pick a value for k
2) for a new observation, find classes of k-nearest neighbors (using an established and consistent distance metric)
3) assign the most-common neighbor class to our new point

Advantages:
- simple to understand and explain
- fast to train
- nonparametric (does not assume some preexisting pattern to classes; just evaluates based on the algo)

Disadvantages:
- prediction phase is slow when n gets very large
- very sensitive to irrelevant features (in terms of speed ?)
