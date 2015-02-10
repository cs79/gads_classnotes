## Class 13: Bayesian Statistics / Naive Bayes

### Probability and Bayes' Theorem

**Sample space:** universe of all possible outcomes
**Event:** the occurrence of a discrete outcome within a sample space

#### Probability slide 7 (overlapping circles diagram):

- A:        have cancer, did not test positive
- AB:       have cancer, tested positive
- B:        do not have cancer, tested positive
- White:    Do not have cancer, did not test positive

#### Slide 8 (confusion matrix):

            Pred                    Pred
            0   1                   0   1
Truth    0  65  10      Truth   0   TN  FP
         1  5   20              1   FN  TP

Conditional probability of cancer | tested positive = 2/3 (20/30)

#### Conditional Probability

P(A|B) = P(AB) / P(B)   # where P(AB) is probability of A intersect B

Conditional probability transforms the sample space (denominators of calculations of P(AB) and P(B) will wash out, resolving a more specific sample space)

#### Deriving Bayes' Theorem

We know:    P(A|B) = P(AB) / P(B) and P(B|A) = P(AB) / P(A)

Thus:       P(AB) = P(A|B) * P(B) = P(B|A) * P(A)

Finally:    **P(A|B) = (P(B|A) * P(A)) / P(B)**

This is Bayes' Theorem.

#### Naive Bayes for classification in Python:

```python

'''
CLASS: Applying Bayes' theorem to iris classification
'''

# load the iris data
from sklearn.datasets import load_iris
iris = load_iris()

# round up the measurements
import numpy as np
X = np.ceil(iris.data)

# clean up column names
features = [name[:-5].replace(' ', '_') for name in iris.feature_names]

# read into pandas
import pandas as pd
df = pd.DataFrame(X, columns=features)

# create a list of species using iris.target and iris.target_names
species = [iris.target_names[num] for num in iris.target]

# add the species list as a new DataFrame column
df['species'] = species

# print the DataFrame
df

# show all observations with features: 7, 3, 5, 2
df[(df.sepal_length==7) & (df.sepal_width==3) & (df.petal_length==5) & (df.petal_width==2)]

## Probability of versicolor given this obs:
prob = ((float(sum(df[(df.sepal_length==7) & (df.sepal_width==3) & (df.petal_length==5) \
& (df.petal_width==2)]['species'] == 'versicolor'))/len(df[df.species=='versicolor'])) *\
len(df[df.species=='versicolor'])/len(df)) / (len(df[(df.sepal_length==7) &             \
(df.sepal_width==3) & (df.petal_length==5) & (df.petal_width==2)])/float(len(df)))

```
#### Practical upshot of Bayes' Theorem in machine learning classification problems

Bayes' Theorem can help us to determine the probability of a record belonging to a class, given the data that we actually observe.

Generically:

P(class C | {x_i}) = (P({x_i} | class C) * P(class C)) / P({x_i})

P(class C)          :   known as the **prior** / prior probabilitiy
P({x_i} | class C)  :   known as the **likelihood function**
P({x_i})            :   known as the **normalization constant**
P(class C | {x_i})  :   known as the **posterior probabilitiy**

### Bayesian Inference

We want to **update** our beliefs about the distribution of C using the data ('evidence') at our disposal.

The likelihood function is the hardest thing to estimate in practice; having an EXACT value would require having ALL THE DATA, which we don't have, but we can use a large enough amount of data to make a reasonable, statistically-underpinned estimate of it.

### Naive Bayes

Simplifying assumption: assume features x_i are independent, so that we can multiply their individual probabilities given class C one by one.

#### Naive Bayes classification

**Training phase:** computes the likelihood function, which is the conditional probability of each feature given each class

**Prediction phase:** computes the posterior probability of each class given the observed features, and choosing the class with the highest probability

If you compute probabilities for 2 classes, and ONLY compute the numerator, you don't even need to compute the normalization constant because it is the common denominator for both -- simply comparing the numerators will give you the same insight about relative likelihood of the different classes.

### SMS Spam Collection example

```python

'''
CLASS: Naive Bayes SMS spam classifier using sklearn
Data source: https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection
'''

## READING IN THE DATA

# read tab-separated file using pandas
import pandas as pd
df = pd.read_table('https://raw.githubusercontent.com/justmarkham/DAT4/master/data/sms.tsv',
                   sep='\t', header=None, names=['label', 'msg'])

# examine the data
df.head(30)
df.label.value_counts()
df.msg.describe()

# convert label to a binary variable
df['label'] = df.label.map({'ham':0, 'spam':1})
df.head()

# split into training and testing sets
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.msg, df.label, random_state=1)
X_train.shape
X_test.shape


## COUNTVECTORIZER: 'convert text into a matrix of token counts'
## http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html

from sklearn.feature_extraction.text import CountVectorizer

# start with a simple example
train_simple = ['call you tonight',
                'Call me a cab',
                'please call me... PLEASE!']

# learn the 'vocabulary' of the training data
vect = CountVectorizer()        # initialize our vectorizer
vect.fit(train_simple)          # run a simple count using all defaults
vect.get_feature_names()        # check what features it found

# transform training data into a 'document-term matrix'
train_simple_dtm = vect.transform(train_simple)         # transform returns something
train_simple_dtm                                        # shows it in sparse format
train_simple_dtm.toarray()                              # shows in non-sparse format

# examine the vocabulary and document-term matrix together
pd.DataFrame(train_simple_dtm.toarray(), columns=vect.get_feature_names())

# transform testing data into a document-term matrix (using existing vocabulary)
test_simple = ["please don't call me"]
test_simple_dtm = vect.transform(test_simple)       # this is using our SAME vectorizer
test_simple_dtm.toarray()                           # returns a 1x6 matrix
pd.DataFrame(test_simple_dtm.toarray(), columns=vect.get_feature_names())


## REPEAT PATTERN WITH SMS DATA

# instantiate the vectorizer
vect = CountVectorizer()

# learn vocabulary and create document-term matrix in a single step
train_dtm = vect.fit_transform(X_train)
train_dtm

# transform testing data into a document-term matrix
test_dtm = vect.transform(X_test)
test_dtm

# store feature names and examine them
train_features = vect.get_feature_names()
len(train_features)
train_features[:50]
train_features[-50:]

# convert train_dtm to a regular array
train_arr = train_dtm.toarray()
train_arr


## MODEL BUILDING WITH NAIVE BAYES
## http://scikit-learn.org/stable/modules/naive_bayes.html

# train a Naive Bayes model using train_dtm
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb.fit(train_dtm, y_train)

# make predictions on test data using test_dtm
preds = nb.predict(test_dtm)
preds

# compare predictions to true labels
from sklearn import metrics
print metrics.accuracy_score(y_test, preds)
print metrics.confusion_matrix(y_test, preds)

# predict (poorly calibrated) probabilities and calculate AUC
probs = nb.predict_proba(test_dtm)[:, 1]
probs
print metrics.roc_auc_score(y_test, probs)

# exercise: show the message text for the false positives
X_test[y_test - preds == -1]
X_test[y_test < preds]          # simpler, same effect

# exercise: show the message text for the false negatives
X_test[y_test - preds == 1]
X_test[y_test > preds]


## COMPARE NAIVE BAYES AND LOGISTIC REGRESSION
## USING ALL DATA AND CROSS-VALIDATION

# create a document-term matrix using all data
all_dtm = vect.fit_transform(df.msg)

# instantiate logistic regression
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()

# compare AUC using cross-validation
# note: this is slightly improper cross-validation... can you figure out why?
# because we trained on ALL corpora; we will never see any "unknown" words in CV
# 100% of them are in our model's vocabulary
from sklearn.cross_validation import cross_val_score
cross_val_score(nb, all_dtm, df.label, cv=10, scoring='roc_auc').mean()
cross_val_score(logreg, all_dtm, df.label, cv=10, scoring='roc_auc').mean()

```
A document-term matrix has rows of documents with columns of terms; often a sparse matrix.

CountVectorizer.fit() learns a vocabulary, CountVectorizer.transform() transforms documents into a document-term matrix (using the vocabulary that it learned from fit())
