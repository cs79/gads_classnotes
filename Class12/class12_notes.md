## Class 12: Cluster Analysis

**Clusters**: groups of similar data points; depends on measure of similarity how they will be clustered by an algorithm.

Cluster analysis is used to divide data into like groups.  Interpretation of those groups comes from a human, not the algos.

### K-Means algo:

1. Choose initial centroids
    - randomly (common)
    - alternative clustering task, use resulting centroids as initial k-means centroids
    - start w/ global centroid, choose point at max distance, repeat (but might select outlier)

2. Assess similarity
    - Choose similarity metric (Euclidean distance is most common)
        - d(x1, x2) = sqrt(sum_i_N((x1i - x2i)^2))
    - Calculate distance from each point to each centroid
    - assign each point to the centroid that it is closest to

3. Recompute centroids
    - calculate based on geometric center of the points assigned to a centroid (by virtue of being closest to it on this iteration)
    - move the centroid to the computed "average" location for its assigned cluster on this iteration of the loop

4. Convergence
    - iterate steps 2 and 3 until reasonable convergence is achieved
        - can use a max_iterations stopping criterion to keep this from running forever if something weird goes wrong

#### Output of K-Means

Will generally converge to k clusters even if no natural clusters exist in the data.  A human will need to interpret the clusters and give them labels if appropriate (either notional or actual data labels).

Two validation metrics for K-means: cohesion, separation

#### Cohesion

Cohesion measures clustering effectiveness **within** a cluster:

C_hat(C_i) = sum_x_in_C_i(d(x, c_i))

You want to minimize this sum of these distances.

#### Separation

Separation measures clustering effectiveness **between** clusters:

S_hat(C_i, C_j) = d(c_i, c_j)

Distance between the centroids -- how far apart are the two clusters?  You want to maximize this distance.

#### Silhouette Coefficient

Combines the ideas of cohesion and separation.  For a point x_j:

SC_i = (b_i - a_i) / max(a_i, b_i)

such that:
    - a_i = average in-cluster distance to x_j          # separation
    - b_ij = average between-cluster distance to x_j
    - b_i = min_j(b_ij)                                 # cohesion

Can take values between -1 and 1.

In general, we want separation to be high and cohesion to be low.  When this is the case, SC is close to 1.

A negative silhouette coefficient means that the cluster radius is larger than the space between them.  This is generally a bad thing, and indicating that clusters are overlapping.

The silhouette coefficient for an entire cluster is the average SC of points within that cluster:

SC(C_i) = 1/m_i * sum_x_in_C_i(SC_i)

The overall silhouette coefficient is given by the average silhouette coefficient across all clusters:

SC_total = 1/k * sum_1_k(SC(C_i))

### Using Silhouette Coefficient

Can be used to determine the best number of clusters for your dataset.  

    - Compute SEE or SC for different values of K; see what value generates the best score

Ultimately, a human has to step in and decide how many clusters to use.  SC is a tool that can help choose this, but the algorithm doesn't know "what" the clusters are.  They are just notional and have "meaning" only when labeled/interpreted.

### K-Means strengths / weaknesses

**Strengths**: computationally efficient, simple, intuititive in nature

**Weaknesses**: highly scale-dependent; not suitable for data with widely varying shapes and densities

### Working with K-means in Python:

```python

from sklearn.cluster import KMeans  # import our desired estimator
import matplotlib.pyplot as plt
from sklearn import datasets
import pandas as pd
import numpy as np

%matplotlib qt

# ------------------------------------------
# EXERCISE: Compute the centoid of the following data
#           [2, 5], [4, 4], [3, 3]
# ------------------------------------------

d = np.array([[2, 5], [4, 4], [3, 3]])
x, y = d.mean(axis=0)                   # axis = 0 reduces over the rows, down the cols
                                        # meaning we get 2 means, 1 per col; unpack tuple
# Import iris data
iris = datasets.load_iris()
d = iris.data

np.random.seed(0)

# Run KMeans
est = KMeans(n_clusters=3, init='random')   # instantiate our estimator
est.fit(d)                                  # "fit" our model (to k centroids)
y_kmeans = est.predict(d)                   # show clusters (fitted) for each observation

# by default, this will pre-compute distances (between points), which saves time but costs memory

# Make a pretty graph:
colors = np.array(['#FF0054','#FBD039','#23C2BC'])
plt.figure()
plt.scatter(d[:, 2], d[:, 0], c=colors[y_kmeans], s=50)     # colors by the array above, as mapped to the "predicted" clusters
plt.xlabel(iris.feature_names[2])
plt.ylabel(iris.feature_names[0])

# ------------------------------------------
# EXERCISE: Find the centers and plot them
#           on the same graph.
# ------------------------------------------

centers = est.cluster_centers_
plt.scatter(centers[:, 2], centers[:, 0], c='k', linewidths=3,  # plot the 3 centers
            marker='+', s=300)                                  # of the 2 relevant cols

'''
VISUALIZING THE CLUSTERS
What are some different options to visualize
multi-dimensional data? Let's look at three ways you can do this.
- Scatter Plot Grid
- 3D Plot
- Parallel Coordinates
'''

#================================
# Option #1: Scatter Plot Grid
plt.figure(figsize=(8, 8))
plt.suptitle('Scatter Plot Grid',  fontsize=14)
# Upper Left
plt.subplot(221)
plt.scatter(d[:,2], d[:,0], c = colors[y_kmeans])
plt.ylabel(iris.feature_names[0])

# Upper Right
plt.subplot(222)
plt.scatter(d[:,3], d[:,0], c = colors[y_kmeans])

# Lower Left
plt.subplot(223)
plt.scatter(d[:,2], d[:,1], c = colors[y_kmeans])
plt.ylabel(iris.feature_names[1])
plt.xlabel(iris.feature_names[2])

# Lower Right
plt.subplot(224)
plt.scatter(d[:,3], d[:,1], c = colors[y_kmeans])
plt.xlabel(iris.feature_names[3])

#================================
# Option #2: 3d plot
from mpl_toolkits.mplot3d import Axes3D
plt.suptitle('3d plot', fontsize=15)
ax = Axes3D(plt.figure(figsize=(10, 9)), rect=[.01, 0, 0.95, 1], elev=30, azim=134)
ax.scatter(d[:,0], d[:,1], d[:,2], c = colors[y_kmeans], s=120)
ax.set_xlabel('Sepal Width')
ax.set_ylabel('Sepal Width')
ax.set_zlabel('Petal Length')
# Modified from the example here:
# http://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_iris.html

# ---------------------------------------
# EXERCISE: Create a Parallel Coordinates
#           visualization with the classes
# ---------------------------------------

#================================
# Option 3: Parallel Coordinates

from pandas.tools.plotting import parallel_coordinates
# I'm going to convert to a pandas dataframe
# Using a snippet of code we learned from one of Kevin's lectures!
features = [name[:-5].title().replace(' ', '') for name in iris.feature_names]
iris_df = pd.DataFrame(iris.data, columns = features)
iris_df['Name'] = iris.target_names[iris.target]
parallel_coordinates(data=iris_df, class_column='Name',
                     colors=('#FF0054', '#FBD039', '#23C2BC'))
```

#### Parallel Coordinates **[SUPER NIFTY]**

A method for easily visualizing multidimensional data.

Each vertical line in the graph is an axis.  Each "string" is a flower.

A flower is no longer a data point; it is a data polyline.  

How to read this graph?

Follow each polyline -- as it travels through the axes, it goes straight to its measurement for each axis.  

If the colored lines seem to "travel together", this is an indication that the groupings represented by the colors will make for good clusters.

In our example, the red is traveling through the dimensions in a different shape.  The blue and yellow seem to be traveling more together, which is reflected in their overlapping 2- and 3-D clusters.

The space in between dimensions is arbitrary -- just evenly spaced to indicate no weighting / make visual analysis more simple.

*This seems awesome.*

### More Python:

```python
'''
DETERMINING THE NUMBER OF CLUSTERS
How do you choose k? There isn't a bright line, but we can evaluate
performance metrics such as the silhouette coefficient and within sum of
squared errors across values of k.

scikit-learn Clustering metrics documentation:
http://scikit-learn.org/stable/modules/classes.html#clustering-metrics
'''

# Create a bunch of different models
k_rng = range(1,15)
est = [KMeans(n_clusters = k).fit(d) for k in k_rng]    # list of instances of fitted
                                                        # k-means estimator objects

#================================
# Option 1: Silhouette Coefficient
# NB: Generally want SC to be closer to 1, while also minimizing k, but not strictly in all cases (e.g. below -- we don't want to actually pick k=2 here)

from sklearn import metrics
silhouette_score = [metrics.silhouette_score(d, e.labels_, metric='euclidean') for e in est[1:]]

# Plot the results
plt.figure(figsize=(7, 8))
plt.subplot(211)
plt.title('Using the elbow method to inform k choice')
plt.plot(k_rng[1:], silhouette_score, 'b*-')
plt.xlim([1,15])
plt.grid(True)
plt.ylabel('Silhouette Coefficient')
plt.plot(3,silhouette_score[1], 'o', markersize=12, markeredgewidth=1.5,
         markerfacecolor='None', markeredgecolor='r')

# -----------------------------------------------------
# EXERCISE: Calculate the within sum of squared errors
#           and plot over a range of k
# -----------------------------------------------------


#================================
# Option 2: Within Sum of Squares (a.k.a., inertia)
# Generally want to minimize WSS, while also minimizing k
# similar to cohesion in the sense that we want to minimize it -- minimize the differences between points and their centroids as k changes

within_sum_squares = [e.inertia_ for e in est]

# Plot the results
plt.subplot(212)
plt.plot(k_rng, within_sum_squares, 'b*-')
plt.xlim([1,15])
plt.grid(True)
plt.xlabel('k')
plt.ylabel('Within Sum of Squares')
plt.plot(3,within_sum_squares[2], 'ro', markersize=12, markeredgewidth=1.5,
      markerfacecolor='None', markeredgecolor='r')

```

### Downsides to K-Means clustering

```python
'''
NOTES ON LIMITATIONS OF K-MEANS CLUSTERING

Adapted from Bart Baddely's 2014 PyData Presentation:
http://nbviewer.ipython.org/github/BartBaddeley/PyDataTalk-2014/blob/master/PyDataTalk.ipynb

Agenda:
1) K-means might not work when dimensions have different scales
2) K-means might not work for non-spherical shapes
3) K-means might not work for clusters of different sizes
'''

from sklearn.cluster import KMeans
import numpy as np
from sklearn.datasets.samples_generator import make_blobs, make_moons
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances

'''
1) DIMENSIONS WITH DIFFERENT SCALES
'''

# Generate data with differing variances
np.random.seed(0)

centres = [[1, 0.75], [1, -0.75], [0, 0]]

X0, labels0_true = make_blobs(n_samples=300, centers=centres[0], cluster_std=[[0.6,0.1]])
X1, labels1_true = make_blobs(n_samples=300, centers=centres[1], cluster_std=[[0.6,0.1]])
X2, labels2_true = make_blobs(n_samples=300, centers=centres[2], cluster_std=[[0.6,0.1]])
X = np.concatenate((X0,X1,X2))
labels_true = np.concatenate((labels0_true,labels1_true+1,labels2_true+2))

colors = np.array(['#FF0054','#FBD039','#23C2BC'])

plt.figure(figsize=(12, 6))
plt.suptitle('Dimensions with Different Scales', fontsize=15)
plt.subplot(121)
for k, col in zip(range(3), colors):
    my_members = labels_true == k
    cluster_center = centres[k]
    plt.scatter(X[my_members, 0], X[my_members, 1], c=col, marker='o',s=20)
    plt.scatter(cluster_center[0], cluster_center[1], c=col, marker='o', s=200)
plt.axis('equal')
plt.title('Original data')

# Compute clustering with 3 Clusters
k_means_3 = KMeans(init='k-means++', n_clusters=3, n_init=10)
k_means_3.fit(X)
k_means_3_labels = k_means_3.labels_
k_means_3_cluster_centres = k_means_3.cluster_centers_

# Plot result
distance = euclidean_distances(k_means_3_cluster_centres,
                               centres,
                               squared=True)
order = distance.argmin(axis=0)
plt.subplot(122)
for k, col in zip(range(3), colors):
    my_members = k_means_3_labels == order[k]
    plt.scatter(X[my_members, 0], X[my_members, 1],c=col, marker='o', s=20)
    cluster_center = k_means_3_cluster_centres[order[k]]
    plt.scatter(cluster_center[0], cluster_center[1], marker = 'o', c=col, s=200, alpha=0.8)
plt.axis('equal')
plt.title('KMeans 3')

## Scales are throwing off proper classification in this example; could resolve with scaling

'''
#2: NON-SPHERICAL SHAPES
'''

[X, true_labels] = make_moons(n_samples=1000, noise=.05)

plt.figure(figsize=(12, 6))
plt.suptitle('Non-Spherical Shapes', fontsize=15)
plt.subplot(121)
for k, col in zip(range(2), colors):
    my_members = true_labels == k
    plt.scatter(X[my_members, 0], X[my_members, 1], c=col, marker='o', s=20)

plt.axis('equal')
plt.title('Original Data')

# Compute clustering with 2 Clusters
k_means_2 = KMeans(init='k-means++', n_clusters=2, n_init=10)
k_means_2.fit(X)
k_means_2_labels = k_means_2.labels_
k_means_2_cluster_centers = k_means_2.cluster_centers_

plt.subplot(122)
for k, col in zip(range(2), colors):
    my_members = k_means_2_labels == k
    plt.scatter(X[my_members, 0], X[my_members, 1],c=col, marker='o', s=20)
    cluster_center = k_means_2_cluster_centers[k]
    plt.scatter(cluster_center[0], cluster_center[1], marker = 'o', c=col, s=200, alpha=0.8)
plt.axis('equal')
plt.title('KMeans 2')

## Despite "obvious" separation when a human looks at it, k-means can't figure out boundaries
## Increasing k will help ameliorate this issue, but may likely not be what you want

'''
#3: CLUSTERS OF DIFFERENT SIZES
'''

np.random.seed(0)

centres = [[-1, 0], [1, 0], [3, 0]]

X0, labels0_true = make_blobs(n_samples=100, centers=centres[0], cluster_std=[[0.2,0.2]])
X1, labels1_true = make_blobs(n_samples=400, centers=centres[1], cluster_std=[[0.6,0.6]])
X2, labels2_true = make_blobs(n_samples=100, centers=centres[2], cluster_std=[[0.2,0.2]])
X = np.concatenate((X0,X1,X2))
labels_true = np.concatenate((labels0_true,labels1_true+1,labels2_true+2))

plt.figure(figsize=(12, 6))
plt.suptitle('Clusters of Different Sizes', fontsize=15)
plt.subplot(121)
for k, col in zip(range(3), colors):
    my_members = labels_true == k
    cluster_center = centres[k]
    plt.scatter(X[my_members, 0], X[my_members, 1], c=col, marker='o',s=20)
    plt.scatter(cluster_center[0], cluster_center[1], c=col, marker='o', s=200)
plt.axis('equal')
plt.title('Original data')

# Compute clustering with 3 Clusters
k_means_3 = KMeans(init='k-means++', n_clusters=3, n_init=10)
k_means_3.fit(X)
k_means_3_labels = k_means_3.labels_
k_means_3_cluster_centres = k_means_3.cluster_centers_

# Plot result
distance = euclidean_distances(k_means_3_cluster_centres,
                               centres,
                               squared=True)
order = distance.argmin(axis=0)
plt.subplot(122)
for k, col in zip(range(3), colors):
    my_members = k_means_3_labels == order[k]
    plt.scatter(X[my_members, 0], X[my_members, 1],c=col, marker='o', s=20)
    cluster_center = k_means_3_cluster_centres[order[k]]
    plt.scatter(cluster_center[0], cluster_center[1], marker = 'o', c=col, s=200, alpha=0.8)
plt.axis('equal')
plt.title('KMeans 3')

## Separation "bleeds" into the center here when it shouldn't, as centroids try to capture spherical shapes

```
