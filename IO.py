data = u"1, 2, 3\n4, 5, 6"

np.genfromtxt(StringIO(data), delimiter=",")
array([[1.,  2.,  3.],
       [4.,  5.,  6.]])




data = u"  1  2  3\n  4  5 67\n890123  4"

np.genfromtxt(StringIO(data), delimiter=3)
array([[  1.,    2.,    3.],
       [  4.,    5.,   67.],
       [890.,  123.,    4.]])

data = u"123456789\n   4  7 9\n   4567 9"

np.genfromtxt(StringIO(data), delimiter=(4, 3, 2))
array([[1234.,   567.,    89.],
       [   4.,     7.,     9.],
       [   4.,   567.,     9.]])




data = u"1, abc , 2\n 3, xxx, 4"

# Without autostrip

np.genfromtxt(StringIO(data), delimiter=",", dtype="|U5")
array([['1', ' abc ', ' 2'],
       ['3', ' xxx', ' 4']], dtype='<U5')

# With autostrip

np.genfromtxt(StringIO(data), delimiter=",", dtype="|U5", autostrip=True)
array([['1', 'abc', '2'],
       ['3', 'xxx', '4']], dtype='<U5')





data = u"""#

# Skip me !

# Skip me too !

1, 2

3, 4

5, 6 #This is the third line of the data

7, 8

# And here comes the last line

9, 0

"""

np.genfromtxt(StringIO(data), comments="#", delimiter=",")
array([[1., 2.],
       [3., 4.],
       [5., 6.],
       [7., 8.],
       [9., 0.]])






data = u"\n".join(str(i) for i in range(10))

np.genfromtxt(StringIO(data),)
array([0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.])

np.genfromtxt(StringIO(data),

              skip_header=3, skip_footer=5)
array([3.,  4.])





data = u"1 2 3\n4 5 6"

np.genfromtxt(StringIO(data), usecols=(0, -1))
array([[1.,  3.],
       [4.,  6.]])





data = u"1 2 3\n4 5 6"

np.genfromtxt(StringIO(data),

              names="a, b, c", usecols=("a", "c"))
array([(1., 3.), (4., 6.)], dtype=[('a', '<f8'), ('c', '<f8')])

np.genfromtxt(StringIO(data),

              names="a, b, c", usecols=("a, c"))
    array([(1., 3.), (4., 6.)], dtype=[('a', '<f8'), ('c', '<f8')])
    
    
    
    
    
    data = StringIO("1 2 3\n 4 5 6")

np.genfromtxt(data, dtype=[(_, int) for _ in "abc"])
array([(1, 2, 3), (4, 5, 6)],
      dtype=[('a', '<i8'), ('b', '<i8'), ('c', '<i8')])




data = StringIO("1 2 3\n 4 5 6")

np.genfromtxt(data, names="A, B, C")
array([(1., 2., 3.), (4., 5., 6.)],
      dtype=[('A', '<f8'), ('B', '<f8'), ('C', '<f8')])



data = StringIO("So it goes\n#a b c\n1 2 3\n 4 5 6")

np.genfromtxt(data, skip_header=1, names=True)
array([(1., 2., 3.), (4., 5., 6.)],
      dtype=[('a', '<f8'), ('b', '<f8'), ('c', '<f8')])




data = StringIO("1 2 3\n 4 5 6")

ndtype=[('a',int), ('b', float), ('c', int)]

names = ["A", "B", "C"]

np.genfromtxt(data, names=names, dtype=ndtype)
array([(1, 2., 3), (4, 5., 6)],
      dtype=[('A', '<i8'), ('B', '<f8'), ('C', '<i8')])



data = StringIO("1 2 3\n 4 5 6")

np.genfromtxt(data, dtype=(int, float, int))
array([(1, 2., 3), (4, 5., 6)],
      dtype=[('f0', '<i8'), ('f1', '<f8'), ('f2', '<i8')])



data = StringIO("1 2 3\n 4 5 6")

np.genfromtxt(data, dtype=(int, float, int), names="a")
array([(1, 2., 3), (4, 5., 6)],
      dtype=[('a', '<i8'), ('f0', '<f8'), ('f1', '<i8')])



data = StringIO("1 2 3\n 4 5 6")

np.genfromtxt(data, dtype=(int, float, int), defaultfmt="var_%02i")
array([(1, 2., 3), (4, 5., 6)],
      dtype=[('var_00', '<i8'), ('var_01', '<f8'), ('var_02', '<i8')])



convertfunc = lambda x: float(x.strip(b"%"))/100.

data = u"1, 2.3%, 45.\n6, 78.9%, 0"

names = ("i", "p", "n")

# General case .....

np.genfromtxt(StringIO(data), delimiter=",", names=names)
array([(1., nan, 45.), (6., nan, 0.)],
      dtype=[('i', '<f8'), ('p', '<f8'), ('n', '<f8')])



# Converted case ...

np.genfromtxt(StringIO(data), delimiter=",", names=names,

              converters={1: convertfunc})
array([(1., 0.023, 45.), (6., 0.789, 0.)],
      dtype=[('i', '<f8'), ('p', '<f8'), ('n', '<f8')])


# Using a name for the converter ...

np.genfromtxt(StringIO(data), delimiter=",", names=names,

              converters={"p": convertfunc})
array([(1., 0.023, 45.), (6., 0.789, 0.)],
      dtype=[('i', '<f8'), ('p', '<f8'), ('n', '<f8')])


data = u"1, , 3\n 4, 5, 6"

convert = lambda x: float(x.strip() or -999)

np.genfromtxt(StringIO(data), delimiter=",",

              converters={1: convert})
array([[   1., -999.,    3.],
       [   4.,    5.,    6.]])



data = u"N/A, 2, 3\n4, ,???"

kwargs = dict(delimiter=",",

              dtype=int,

              names="a,b,c",

              missing_values={0:"N/A", 'b':" ", 2:"???"},

              filling_values={0:0, 'b':0, 2:-999})

np.genfromtxt(StringIO(data), **kwargs)
array([(0, 2, 3), (4, 0, -999)],
      dtype=[('a', '<i8'), ('b', '<i8'), ('c', '<i8')])



