x = np.arange(10)

x[2]
2

x[-2]
8




x.shape = (2, 5)  # now x is 2-dimensional

x[1, 3]
8

x[1, -1]
9




x[0]
array([0, 1, 2, 3, 4])




x[0][2]
2




x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

x[1:7:2]
array([1, 3, 5])




x[-2:10]
array([8, 9])

x[-3:3:-1]
array([7, 6, 5, 4])




x[5:]
array([5, 6, 7, 8, 9])



x = np.array([[[1],[2],[3]], [[4],[5],[6]]])

x.shape
(2, 3, 1)

x[1:2]
array([[[4],
        [5],
        [6]]])





x[..., 0]
array([[1, 2, 3],
      [4, 5, 6]])




x[:, :, 0]
array([[1, 2, 3],
      [4, 5, 6]])



x[:, np.newaxis, :, :].shape
(2, 1, 3, 1)

x[:, None, :, :].shape
(2, 1, 3, 1)



x = np.arange(5)

x[:, np.newaxis] + x[np.newaxis, :]
array([[0, 1, 2, 3, 4],
      [1, 2, 3, 4, 5],
      [2, 3, 4, 5, 6],
      [3, 4, 5, 6, 7],
      [4, 5, 6, 7, 8]])




x = np.arange(10, 1, -1)

x
array([10,  9,  8,  7,  6,  5,  4,  3,  2])

x[np.array([3, 3, 1, 8])]
array([7, 7, 9, 2])

x[np.array([3, 3, -3, 8])]
array([7, 7, 4, 2])



x = np.array([[1, 2], [3, 4], [5, 6]])

x[np.array([1, -1])]
array([[3, 4],
      [5, 6]])

x[np.array([3, 4])]
Traceback (most recent call last):
    
    
    
    
    y = np.arange(35).reshape(5, 7)

y
array([[ 0,  1,  2,  3,  4,  5,  6],
       [ 7,  8,  9, 10, 11, 12, 13],
       [14, 15, 16, 17, 18, 19, 20],
       [21, 22, 23, 24, 25, 26, 27],
       [28, 29, 30, 31, 32, 33, 34]])

y[np.array([0, 2, 4]), np.array([0, 1, 2])]
array([ 0, 15, 30])






y[np.array([0, 2, 4]), np.array([0, 1])]
Traceback (most recent call last):
  ...
IndexError: shape mismatch: indexing arrays could not be broadcast
together with shapes (3,) (2,)





x = np.array([[ 0,  1,  2],

              [ 3,  4,  5],

              [ 6,  7,  8],

              [ 9, 10, 11]])

rows = np.array([[0, 0],

                 [3, 3]], dtype=np.intp)

columns = np.array([[0, 2],

                    [0, 2]], dtype=np.intp)

x[rows, columns]
array([[ 0,  2],
       [ 9, 11]])






rows = np.array([0, 3], dtype=np.intp)

columns = np.array([0, 2], dtype=np.intp)

rows[:, np.newaxis]
array([[0],
       [3]])

x[rows[:, np.newaxis], columns]
array([[ 0,  2],
       [ 9, 11]])





x[np.ix_(rows, columns)]
array([[ 0,  2],
       [ 9, 11]])



x[rows, columns]
array([ 0, 11])



x = np.array([[1., 2.], [np.nan, 3.], [np.nan, np.nan]])

x[~np.isnan(x)]
array([1., 2., 3.])



x = np.array([1., -1., -2., 3])

x[x < 0] += 20

x
array([ 1., 19., 18., 3.])



x = np.arange(35).reshape(5, 7)

b = x > 20

b[:, 5]
array([False, False, False,  True,  True])

x[b[:, 5]]
array([[21, 22, 23, 24, 25, 26, 27],
      [28, 29, 30, 31, 32, 33, 34]])




x = np.array([[0, 1], [1, 1], [2, 2]])

rowsum = x.sum(-1)

x[rowsum <= 2, :]
array([[0, 1],
       [1, 1]])




x = np.array([[ 0,  1,  2],

              [ 3,  4,  5],

              [ 6,  7,  8],

              [ 9, 10, 11]])

rows = (x.sum(-1) % 2) == 0

rows
array([False,  True, False,  True])

columns = [0, 2]

x[np.ix_(rows, columns)]
array([[ 3,  5],
       [ 9, 11]])



rows = rows.nonzero()[0]

x[rows[:, np.newaxis], columns]
array([[ 3,  5],
       [ 9, 11]])



x = np.arange(30).reshape(2, 3, 5)

x
array([[[ 0,  1,  2,  3,  4],
        [ 5,  6,  7,  8,  9],
        [10, 11, 12, 13, 14]],
      [[15, 16, 17, 18, 19],
        [20, 21, 22, 23, 24],
        [25, 26, 27, 28, 29]]])

b = np.array([[True, True, False], [False, True, True]])

x[b]
array([[ 0,  1,  2,  3,  4],
      [ 5,  6,  7,  8,  9],
      [20, 21, 22, 23, 24],
      [25, 26, 27, 28, 29]])




y = np.arange(35).reshape(5,7)

y[np.array([0, 2, 4]), 1:3]
array([[ 1,  2],
       [15, 16],
       [29, 30]])




y[:, 1:3][np.array([0, 2, 4]), :]
array([[ 1,  2],
       [15, 16],
       [29, 30]])



x = np.array([[ 0,  1,  2],

              [ 3,  4,  5],

              [ 6,  7,  8],

              [ 9, 10, 11]])

x[1:2, 1:3]
array([[4, 5]])

x[1:2, [1, 2]]
array([[4, 5]])




x = np.arange(35).reshape(5, 7)

b = x > 20

b
array([[False, False, False, False, False, False, False],
      [False, False, False, False, False, False, False],
      [False, False, False, False, False, False, False],
      [ True,  True,  True,  True,  True,  True,  True],
      [ True,  True,  True,  True,  True,  True,  True]])

x[b[:, 5], 1:3]
array([[22, 23],
      [29, 30]])



x = np.zeros((2, 2), dtype=[('a', np.int32), ('b', np.float64, (3, 3))])

x['a'].shape
(2, 2)

x['a'].dtype
dtype('int32')

x['b'].shape
(2, 2, 3, 3)

x['b'].dtype
dtype('float64')





x = np.arange(10)

x[2:7] = 1



x[2:7] = np.arange(5)



x[1] = 1.2

x[1]
1

x[1] = 1.2j
Traceback (most recent call last):
  ...
TypeError: can't convert complex to int




x = np.arange(0, 50, 10)

x
array([ 0, 10, 20, 30, 40])

x[np.array([1, 1, 3, 1])] += 1

x
array([ 0, 11, 20, 31, 40])



z = np.arange(81).reshape(3, 3, 3, 3)

indices = (1, 1, 1, 1)

z[indices]
40




indices = (1, 1, 1, slice(0, 2))  # same as [1, 1, 1, 0:2]

z[indices]
array([39, 40])



indices = (1, Ellipsis, 1)  # same as [1, ..., 1]

z[indices]
array([[28, 31, 34],
       [37, 40, 43],
       [46, 49, 52]])


z[[1, 1, 1, 1]]  # produces a large array
array([[[[27, 28, 29],
         [30, 31, 32], ...

z[(1, 1, 1, 1)]  # returns a single value
40



