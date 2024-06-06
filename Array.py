a = np.array([1, 2, 3, 4, 5, 6])
a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
print(a[0])
[1 2 3 4]


import numpy as np

a = np.array([1, 2, 3])


np.zeros(2)
array([0., 0.])


np.ones(2)
array([1., 1.])


# Create an empty array with 2 elements

np.empty(2) 
array([3.14, 42.  ])  # may vary


np.arange(4)
array([0, 1, 2, 3])


np.arange(2, 9, 2)
array([2, 4, 6, 8])


np.linspace(0, 10, num=5)
array([ 0. ,  2.5,  5. ,  7.5, 10. ])



x = np.ones(2, dtype=np.int64)

x
array([1, 1])


arr = np.array([2, 1, 5, 3, 7, 4, 6, 8])


np.sort(arr)
array([1, 2, 3, 4, 5, 6, 7, 8])


a = np.array([1, 2, 3, 4])

b = np.array([5, 6, 7, 8])


np.concatenate((a, b))
array([1, 2, 3, 4, 5, 6, 7, 8])


x = np.array([[1, 2], [3, 4]])

y = np.array([[5, 6]])


np.concatenate((x, y), axis=0)
array([[1, 2],
       [3, 4],
       [5, 6]])


array_example = np.array([[[0, 1, 2, 3],

                           [4, 5, 6, 7]],


                          [[0, 1, 2, 3],

                           [4, 5, 6, 7]],


                          [[0 ,1 ,2, 3],

                           [4, 5, 6, 7]]])


array_example.ndim
3



array_example.size
24


array_example.shape
(3, 2, 4)


a = np.arange(6)

print(a)
[0 1 2 3 4 5]


b = a.reshape(3, 2)

print(b)
[[0 1]
 [2 3]
 [4 5]]


np.reshape(a, newshape=(1, 6), order='C')
array([[0, 1, 2, 3, 4, 5]])



a = np.array([1, 2, 3, 4, 5, 6])

a.shape
(6,)


a2 = a[np.newaxis, :]

a2.shape
(1, 6)


row_vector = a[np.newaxis, :]

row_vector.shape
(1, 6)


col_vector = a[:, np.newaxis]

col_vector.shape
(6, 1)


a = np.array([1, 2, 3, 4, 5, 6])

a.shape
(6,)


b = np.expand_dims(a, axis=1)

b.shape
(6, 1)



c = np.expand_dims(a, axis=0)

c.shape
(1, 6)



data = np.array([1, 2, 3])

data[1]
2

data[0:2]
array([1, 2])

data[1:]
array([2, 3])

data[-2:]
array([2, 3])


a = np.array([[1 , 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])


print(a[a < 5])
[1 2 3 4]


five_up = (a >= 5)

print(a[five_up])
[ 5  6  7  8  9 10 11 12]


divisible_by_2 = a[a%2==0]

print(divisible_by_2)
[ 2  4  6  8 10 12]


c = a[(a > 2) & (a < 11)]

print(c)
[ 3  4  5  6  7  8  9 10]


five_up = (a > 5) | (a == 5)

print(five_up)
[[False False False False]
 [ True  True  True  True]
 [ True  True  True True]]


a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])


b = np.nonzero(a < 5)

print(b)
(array([0, 0, 0, 0]), array([0, 1, 2, 3]))


list_of_coordinates= list(zip(b[0], b[1]))

for coord in list_of_coordinates:

    print(coord)
(0, 0)
(0, 1)
(0, 2)
(0, 3)


print(a[b])
[1 2 3 4]



not_there = np.nonzero(a == 42)

print(not_there)
(array([], dtype=int64), array([], dtype=int64))


a = np.array([1,  2,  3,  4,  5,  6,  7,  8,  9, 10])


arr1 = a[3:8]

arr1
array([4, 5, 6, 7, 8])


a1 = np.array([[1, 1],

               [2, 2]])

a2 = np.array([[3, 3],

               [4, 4]])


np.vstack((a1, a2))
array([[1, 1],
       [2, 2],
       [3, 3],
       [4, 4]])


np.hstack((a1, a2))
array([[1, 1, 3, 3],
       [2, 2, 4, 4]])


x = np.arange(1, 25).reshape(2, 12)

x
array([[ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12],
       [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]])


np.hsplit(x, 3)
  [array([[ 1,  2,  3,  4],
         [13, 14, 15, 16]]), array([[ 5,  6,  7,  8],
         [17, 18, 19, 20]]), array([[ 9, 10, 11, 12],
         [21, 22, 23, 24]])]
                                    
                                    
                                    np.hsplit(x, (3, 4))
  [array([[ 1,  2,  3],
         [13, 14, 15]]), array([[ 4],
         [16]]), array([[ 5,  6,  7,  8,  9, 10, 11, 12],
         [17, 18, 19, 20, 21, 22, 23, 24]])]
                        
                        
                        a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
                        
                        
                        b1 = a[0, :]

b1
array([1, 2, 3, 4])

b1[0] = 99

b1
array([99,  2,  3,  4])

a
array([[99,  2,  3,  4],
       [ 5,  6,  7,  8],
       [ 9, 10, 11, 12]])




b2 = a.copy()


data = np.array([1, 2])

ones = np.ones(2, dtype=int)

data + ones
array([2, 3])


data - ones
array([0, 1])

data * data
array([1, 4])

data / data
array([1., 1.])


a = np.array([1, 2, 3, 4])

a.sum()
10


b = np.array([[1, 1], [2, 2]])



b.sum(axis=0)
array([3, 3])


b.sum(axis=1)
array([2, 4])


ata = np.array([1.0, 2.0])

data * 1.6
array([1.6, 3.2])


a.sum()
4.8595784


a.min()
0.05093587


a.min(axis=0)
array([0.12697628, 0.05093587, 0.26590556, 0.5510652 ])


data = np.array([[1, 2], [3, 4], [5, 6]])

data
array([[1, 2],
       [3, 4],
       [5, 6]])


data[0, 1]
2

data[1:3]
array([[3, 4],
       [5, 6]])

data[0:2, 0]
array([1, 3])


data.max()
6

data.min()
1

data.sum()
21


data = np.array([[1, 2], [5, 3], [4, 6]])

data
array([[1, 2],
       [5, 3],
       [4, 6]])

data.max(axis=0)
array([5, 6])

data.max(axis=1)
array([2, 5, 6])


data = np.array([[1, 2], [3, 4]])

ones = np.array([[1, 1], [1, 1]])

data + ones
array([[2, 3],
       [4, 5]])



data = np.array([[1, 2], [3, 4], [5, 6]])

ones_row = np.array([[1, 1]])

data + ones_row
array([[2, 3],
       [4, 5],
       [6, 7]])


np.ones((4, 3, 2))
array([[[1., 1.],
        [1., 1.],
        [1., 1.]],

       [[1., 1.],
        [1., 1.],
        [1., 1.]],

       [[1., 1.],
        [1., 1.],
        [1., 1.]],

       [[1., 1.],
        [1., 1.],
        [1., 1.]]])


np.ones(3)
array([1., 1., 1.])

np.zeros(3)
array([0., 0., 0.])

rng = np.random.default_rng()  # the simplest way to generate random numbers

rng.random(3) 
array([0.63696169, 0.26978671, 0.04097352])


np.ones((3, 2))
array([[1., 1.],
       [1., 1.],
       [1., 1.]])

np.zeros((3, 2))
array([[0., 0.],
       [0., 0.],
       [0., 0.]])

rng.random((3, 2)) 
array([[0.01652764, 0.81327024],
       [0.91275558, 0.60663578],
       [0.72949656, 0.54362499]])  # may vary


rng.integers(5, size=(2, 4)) 
array([[2, 1, 1, 0],
       [0, 0, 0, 4]])  # may vary


a = np.array([11, 11, 12, 13, 14, 15, 16, 17, 12, 13, 11, 14, 18, 19, 20])


unique_values = np.unique(a)

print(unique_values)
[11 12 13 14 15 16 17 18 19 20]



unique_values, indices_list = np.unique(a, return_index=True)

print(indices_list)
[ 0  2  3  4  5  6  7 12 13 14]


unique_values, occurrence_count = np.unique(a, return_counts=True)

print(occurrence_count)
[3 2 2 2 1 1 1 1 1 1]


a_2d = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [1, 2, 3, 4]])


unique_values = np.unique(a_2d)

print(unique_values)
[ 1  2  3  4  5  6  7  8  9 10 11 12]


unique_rows = np.unique(a_2d, axis=0)

print(unique_rows)
[[ 1  2  3  4]
 [ 5  6  7  8]
 [ 9 10 11 12]]


unique_rows, indices, occurrence_count = np.unique(

     a_2d, axis=0, return_counts=True, return_index=True)

print(unique_rows)
[[ 1  2  3  4]
 [ 5  6  7  8]
 [ 9 10 11 12]]

print(indices)
[0 1 2]

print(occurrence_count)
[2 1 1]


data.reshape(2, 3)
array([[1, 2, 3],
       [4, 5, 6]])

data.reshape(3, 2)
array([[1, 2],
       [3, 4],
       [5, 6]])


arr = np.arange(6).reshape((2, 3))

arr
array([[0, 1, 2],
       [3, 4, 5]])


arr.transpose()
array([[0, 3],
       [1, 4],
       [2, 5]])


arr.T
array([[0, 3],
       [1, 4],
       [2, 5]])


arr = np.array([1, 2, 3, 4, 5, 6, 7, 8])


reversed_arr = np.flip(arr)


print('Reversed Array: ', reversed_arr)
Reversed Array:  [8 7 6 5 4 3 2 1]


arr_2d = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])


reversed_arr = np.flip(arr_2d)

print(reversed_arr)
[[12 11 10  9]
 [ 8  7  6  5]
 [ 4  3  2  1]]


reversed_arr_rows = np.flip(arr_2d, axis=0)

print(reversed_arr_rows)
[[ 9 10 11 12]
 [ 5  6  7  8]
 [ 1  2  3  4]]


reversed_arr_columns = np.flip(arr_2d, axis=1)

print(reversed_arr_columns)
[[ 4  3  2  1]
 [ 8  7  6  5]
 [12 11 10  9]]


arr_2d[1] = np.flip(arr_2d[1])

print(arr_2d)
[[ 1  2  3  4]
 [ 8  7  6  5]
 [ 9 10 11 12]]


arr_2d[:,1] = np.flip(arr_2d[:,1])

print(arr_2d)
[[ 1 10  3  4]
 [ 8  7  6  5]
 [ 9  2 11 12]]


x = np.array([[1 , 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])


x.flatten()
array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12])


a1 = x.flatten()

a1[0] = 99

print(x)  # Original array
[[ 1  2  3  4]
 [ 5  6  7  8]
 [ 9 10 11 12]]

print(a1)  # New array
[99  2  3  4  5  6  7  8  9 10 11 12]


a2 = x.ravel()

a2[0] = 98

print(x)  # Original array
[[98  2  3  4]
 [ 5  6  7  8]
 [ 9 10 11 12]]

print(a2)  # New array
[98  2  3  4  5  6  7  8  9 10 11 12]


help(max)
Help on built-in function max in module builtins:

max(...)
    max(iterable, *[, default=obj, key=func]) -> value
    max(arg1, arg2, *args, *[, key=func]) -> value

    With a single iterable argument, return its biggest item. The
    default keyword-only argument specifies an object to return if
    the provided iterable is empty.
    With two or more arguments, return the largest argument.
    
    
    max?
max(iterable, *[, default=obj, key=func]) -> value
max(arg1, arg2, *args, *[, key=func]) -> value

With a single iterable argument, return its biggest item. The
default keyword-only argument specifies an object to return if
the provided iterable is empty.
With two or more arguments, return the largest argument.
Type:      builtin_function_or_method



a = np.array([1, 2, 3, 4, 5, 6])



a?
Type:            ndarray
String form:     [1 2 3 4 5 6]
Length:          6
File:            ~/anaconda3/lib/python3.9/site-packages/numpy/__init__.py
Docstring:       <no docstring>
Class docstring:
ndarray(shape, dtype=float, buffer=None, offset=0,
        strides=None, order=None)

An array object represents a multidimensional, homogeneous array
of fixed-size items.  An associated data-type object describes the
format of each element in the array (its byte-order, how many bytes it
occupies in memory, whether it is an integer, a floating point number,
or something else, etc.)

Arrays should be constructed using `array`, `zeros` or `empty` (refer
to the See Also section below).  The parameters given here refer to
a low-level method (`ndarray(...)`) for instantiating an array.

For more information, refer to the `numpy` module and examine the
methods and attributes of an array.

Parameters
----------
(for the __new__ method; see Notes below)

shape : tuple of ints
        Shape of created array.
...


def double(a):

  '''Return a * 2'''

  return a * 2






double?
Signature: double(a)
Docstring: Return a * 2
File:      ~/Desktop/<ipython-input-23-b5adf20be596>
Type:      function





double??
Signature: double(a)
Source:
def double(a):
    '''Return a * 2'''
    return a * 2
File:      ~/Desktop/<ipython-input-23-b5adf20be596>
Type:      function




len?
Signature: len(obj, /)
Docstring: Return the number of items in a container.
Type:      builtin_function_or_method





len??
Signature: len(obj, /)
Docstring: Return the number of items in a container.
Type:      builtin_function_or_method







a = np.array([1, 2, 3, 4, 5, 6])




np.save('filename', a)





b = np.load('filename.npy')




print(b)
[1 2 3 4 5 6]



csv_arr = np.array([1, 2, 3, 4, 5, 6, 7, 8])




np.savetxt('new_file.csv', csv_arr)




np.loadtxt('new_file.csv')
array([1., 2., 3., 4., 5., 6., 7., 8.])




import pandas as pd

# If all of your columns are the same type:

x = pd.read_csv('music.csv', header=0).values

print(x)
[['Billie Holiday' 'Jazz' 1300000 27000000]
 ['Jimmie Hendrix' 'Rock' 2700000 70000000]
 ['Miles Davis' 'Jazz' 1500000 48000000]
 ['SIA' 'Pop' 2000000 74000000]]

# You can also simply select the columns you need:

x = pd.read_csv('music.csv', usecols=['Artist', 'Plays']).values

print(x)
[['Billie Holiday' 27000000]
 ['Jimmie Hendrix' 70000000]
 ['Miles Davis' 48000000]
 ['SIA' 74000000]]










a = np.array([[-2.58289208,  0.43014843, -1.24082018, 1.59572603],

              [ 0.99027828, 1.17150989,  0.94125714, -0.14692469],

              [ 0.76989341,  0.81299683, -0.95068423, 0.11769564],

              [ 0.20484034,  0.34784527,  1.96979195, 0.51992837]])






df = pd.DataFrame(a)

print(df)
          0         1         2         3
0 -2.582892  0.430148 -1.240820  1.595726
1  0.990278  1.171510  0.941257 -0.146925
2  0.769893  0.812997 -0.950684  0.117696
3  0.204840  0.347845  1.969792  0.519928





df.to_csv('pd.csv')





data = pd.read_csv('pd.csv')




np.savetxt('np.csv', a, fmt='%.2f', delimiter=',', header='1,  2,  3,  4')






cat np.csv
#  1,  2,  3,  4
-2.58,0.43,-1.24,1.60
0.99,1.17,0.94,-0.15
0.77,0.81,-0.95,0.12
0.20,0.35,1.97,0.52







a = np.array([2, 1, 5, 7, 4, 6, 8, 14, 10, 9, 18, 20, 22])







import matplotlib.pyplot as plt

# If you're using Jupyter Notebook, you may also want to run the following
# line of code to display your code in the notebook:

%matplotlib inline







plt.plot(a)

# If you are running from a command line, you may need to do this:
# >>> plt.show()







x = np.linspace(0, 5, 20)

y = np.linspace(0, 10, 20)

plt.plot(x, y, 'purple') # line

plt.plot(x, y, 'o')      # dots






fig = plt.figure()

ax = fig.add_subplot(projection='3d')

X = np.arange(-5, 5, 0.15)

Y = np.arange(-5, 5, 0.15)

X, Y = np.meshgrid(X, Y)

R = np.sqrt(X**2 + Y**2)

Z = np.sin(R)

ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis')







