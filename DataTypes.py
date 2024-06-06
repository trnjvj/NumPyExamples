x = np.float32(1.0)

x
1.0

y = np.int_([1,2,4])

y
array([1, 2, 4])

z = np.arange(3, dtype=np.uint8)

z
array([0, 1, 2], dtype=uint8)




np.array([1, 2, 3], dtype='f')
array([1.,  2.,  3.], dtype=float32)



z.astype(float)                 
array([0.,  1.,  2.])

np.int8(z)
array([0, 1, 2], dtype=int8)



z.dtype
dtype('uint8')



d = np.dtype(int)

d 
dtype('int32')

np.issubdtype(d, np.integer)
True

np.issubdtype(d, np.floating)
False



np.power(100, 8, dtype=np.int64)
10000000000000000

np.power(100, 8, dtype=np.int32)
1874919424





np.iinfo(int) # Bounds of the default integer on this system.
iinfo(min=-9223372036854775808, max=9223372036854775807, dtype=int64)

np.iinfo(np.int32) # Bounds of a 32-bit integer
iinfo(min=-2147483648, max=2147483647, dtype=int32)

np.iinfo(np.int64) # Bounds of a 64-bit integer
iinfo(min=-9223372036854775808, max=9223372036854775807, dtype=int64)





np.power(100, 100, dtype=np.int64) # Incorrect even with 64-bit int
0

np.power(100, 100, dtype=np.float64)
1e+200




