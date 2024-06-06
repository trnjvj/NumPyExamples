a1D = np.array([1, 2, 3, 4])

a2D = np.array([[1, 2], [3, 4]])

a3D = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])




a = np.array([127, 128, 129], dtype=np.int8)

a
array([ 127, -128, -127], dtype=int8)




a = np.array([2, 3, 4], dtype=np.uint32)

b = np.array([5, 6, 7], dtype=np.uint32)

c_unsigned32 = a - b

print('unsigned c:', c_unsigned32, c_unsigned32.dtype)
unsigned c: [4294967293 4294967293 4294967293] uint32

c_signed32 = a - b.astype(np.int32)

print('signed c:', c_signed32, c_signed32.dtype)
signed c: [-3 -3 -3] int64




np.arange(10)
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

np.arange(2, 10, dtype=float)
array([2., 3., 4., 5., 6., 7., 8., 9.])

np.arange(2, 3, 0.1)
array([2. , 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9])






np.linspace(1., 4., 6)
array([1. ,  1.6,  2.2,  2.8,  3.4,  4. ])




np.eye(3)
array([[1., 0., 0.],
       [0., 1., 0.],
       [0., 0., 1.]])

np.eye(3, 5)
array([[1., 0., 0., 0., 0.],
       [0., 1., 0., 0., 0.],
       [0., 0., 1., 0., 0.]])





np.diag([1, 2, 3])
array([[1, 0, 0],
       [0, 2, 0],
       [0, 0, 3]])

np.diag([1, 2, 3], 1)
array([[0, 1, 0, 0],
       [0, 0, 2, 0],
       [0, 0, 0, 3],
       [0, 0, 0, 0]])

a = np.array([[1, 2], [3, 4]])

np.diag(a)
array([1, 4])




np.vander(np.linspace(0, 2, 5), 2)
array([[0. , 1. ],
      [0.5, 1. ],
      [1. , 1. ],
      [1.5, 1. ],
      [2. , 1. ]])

np.vander([1, 2, 3, 4], 2)
array([[1, 1],
       [2, 1],
       [3, 1],
       [4, 1]])

np.vander((1, 2, 3, 4), 4)
array([[ 1,  1,  1,  1],
       [ 8,  4,  2,  1],
       [27,  9,  3,  1],
       [64, 16,  4,  1]])





np.zeros((2, 3))
array([[0., 0., 0.],
       [0., 0., 0.]])

np.zeros((2, 3, 2))
array([[[0., 0.],
        [0., 0.],
        [0., 0.]],

       [[0., 0.],
        [0., 0.],
        [0., 0.]]])




np.ones((2, 3))
array([[1., 1., 1.],
       [1., 1., 1.]])

np.ones((2, 3, 2))
array([[[1., 1.],
        [1., 1.],
        [1., 1.]],

       [[1., 1.],
        [1., 1.],
        [1., 1.]]])




from numpy.random import default_rng

default_rng(42).random((2,3))
array([[0.77395605, 0.43887844, 0.85859792],
       [0.69736803, 0.09417735, 0.97562235]])

default_rng(42).random((2,3,2))
array([[[0.77395605, 0.43887844],
        [0.85859792, 0.69736803],
        [0.09417735, 0.97562235]],
       [[0.7611397 , 0.78606431],
        [0.12811363, 0.45038594],
        [0.37079802, 0.92676499]]])





np.indices((3,3))
array([[[0, 0, 0],
        [1, 1, 1],
        [2, 2, 2]],
       [[0, 1, 2],
        [0, 1, 2],
        [0, 1, 2]]])




a = np.array([1, 2, 3, 4, 5, 6])

b = a[:2]

b += 1

print('a =', a, '; b =', b)
a = [2 3 3 4 5 6] ; b = [2 3]




a = np.array([1, 2, 3, 4])

b = a[:2].copy()

b += 1

print('a = ', a, 'b = ', b)
a =  [1 2 3 4] b =  [2 3]





A = np.ones((2, 2))

B = np.eye(2, 2)

C = np.zeros((2, 2))

D = np.diag((-3, -4))

np.block([[A, B], [C, D]])
array([[ 1.,  1.,  1.,  0.],
       [ 1.,  1.,  0.,  1.],
       [ 0.,  0., -3.,  0.],
       [ 0.,  0.,  0., -4.]])





cat simple.csv
x, y
0, 0
1, 1
2, 4
3, 9



np.loadtxt('simple.csv', delimiter = ',', skiprows = 1) 
array([[0., 0.],
       [1., 1.],
       [2., 4.],
       [3., 9.]])




