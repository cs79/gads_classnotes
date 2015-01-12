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






```
