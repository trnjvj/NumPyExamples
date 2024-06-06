a = np.array([1.0, 2.0, 3.0])

b = np.array([2.0, 2.0, 2.0])

a * b
array([2.,  4.,  6.])



a = np.array([1.0, 2.0, 3.0])

b = 2.0

a * b
array([2.,  4.,  6.])





Image  (3d array): 256 x 256 x 3
Scale  (1d array):             3
Result (3d array): 256 x 256 x 3



A      (4d array):  8 x 1 x 6 x 1
B      (3d array):      7 x 1 x 5
Result (4d array):  8 x 7 x 6 x 5



A      (2d array):  5 x 4
B      (1d array):      1
Result (2d array):  5 x 4

A      (2d array):  5 x 4
B      (1d array):      4
Result (2d array):  5 x 4

A      (3d array):  15 x 3 x 5
B      (3d array):  15 x 1 x 5
Result (3d array):  15 x 3 x 5

A      (3d array):  15 x 3 x 5
B      (2d array):       3 x 5
Result (3d array):  15 x 3 x 5

A      (3d array):  15 x 3 x 5
B      (2d array):       3 x 1
Result (3d array):  15 x 3 x 5






A      (1d array):  3
B      (1d array):  4 # trailing dimensions do not match

A      (2d array):      2 x 1
B      (3d array):  8 x 4 x 3 # second from last dimensions mismatched




a = np.array([[ 0.0,  0.0,  0.0],

              [10.0, 10.0, 10.0],

              [20.0, 20.0, 20.0],

              [30.0, 30.0, 30.0]])

b = np.array([1.0, 2.0, 3.0])

a + b
array([[  1.,   2.,   3.],
        [11.,  12.,  13.],
        [21.,  22.,  23.],
        [31.,  32.,  33.]])

b = np.array([1.0, 2.0, 3.0, 4.0])

a + b
Traceback (most recent call last):
ValueError: operands could not be broadcast together with shapes (4,3) (4,)




a = np.array([0.0, 10.0, 20.0, 30.0])

b = np.array([1.0, 2.0, 3.0])

a[:, np.newaxis] + b
array([[ 1.,   2.,   3.],
       [11.,  12.,  13.],
       [21.,  22.,  23.],
       [31.,  32.,  33.]])




from numpy import array, argmin, sqrt, sum

observation = array([111.0, 188.0])

codes = array([[102.0, 203.0],

               [132.0, 193.0],

               [45.0, 155.0],

               [57.0, 173.0]])

diff = codes - observation    # the broadcast happens here

dist = sqrt(sum(diff**2,axis=-1))

argmin(dist)



Observation      (1d array):      2
Codes            (2d array):  4 x 2
Diff             (2d array):  4 x 2




Observation      (2d array):      10 x 3
Codes            (3d array):   5 x 1 x 3
Diff             (3d array):  5 x 10 x 3



