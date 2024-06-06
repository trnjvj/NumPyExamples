x = np.arange(10)

x
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

y = x[1:3]  # creates a view

y
array([1, 2])

x[1:3] = [10, 11]

x
array([ 0, 10, 11,  3,  4,  5,  6,  7,  8,  9])

y
array([10, 11])





x = np.arange(9).reshape(3, 3)

x
array([[0, 1, 2],
       [3, 4, 5],
       [6, 7, 8]])

y = x[[1, 2]]

y
array([[3, 4, 5],
       [6, 7, 8]])

y.base is None




x[[1, 2]] = [[10, 11, 12], [13, 14, 15]]

x
array([[ 0,  1,  2],
       [10, 11, 12],
       [13, 14, 15]])

y
array([[3, 4, 5],
       [6, 7, 8]])




x = np.ones((2, 3))

y = x.T  # makes the array non-contiguous

y
array([[1., 1.],
       [1., 1.],
       [1., 1.]])

z = y.view()

z.shape = 6
Traceback (most recent call last):
   ...
AttributeError: Incompatible shape for in-place modification. Use
`.reshape()` to make a copy with the desired shape.





x = np.arange(9)

x
array([0, 1, 2, 3, 4, 5, 6, 7, 8])

y = x.reshape(3, 3)

y
array([[0, 1, 2],
       [3, 4, 5],
       [6, 7, 8]])

y.base  # .reshape() creates a view
array([0, 1, 2, 3, 4, 5, 6, 7, 8])

z = y[[2, 1]]

z
array([[6, 7, 8],
       [3, 4, 5]])

z.base is None  # advanced indexing creates a copy
True





