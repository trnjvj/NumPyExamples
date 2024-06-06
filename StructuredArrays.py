x = np.array([('Rex', 9, 81.0), ('Fido', 3, 27.0)],

             dtype=[('name', 'U10'), ('age', 'i4'), ('weight', 'f4')])

x
array([('Rex', 9, 81.), ('Fido', 3, 27.)],
      dtype=[('name', '<U10'), ('age', '<i4'), ('weight', '<f4')])




x[1]
('Fido', 3, 27.)



x['age']
array([9, 3], dtype=int32)

x['age'] = 5

x
array([('Rex', 5, 81.), ('Fido', 5, 27.)],
      dtype=[('name', '<U10'), ('age', '<i4'), ('weight', '<f4')])




np.dtype([('x', 'f4'), ('y', np.float32), ('z', 'f4', (2, 2))])
dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4', (2, 2))])




np.dtype([('x', 'f4'), ('', 'i4'), ('z', 'i8')])
dtype([('x', '<f4'), ('f1', '<i4'), ('z', '<i8')])



np.dtype('i8, f4, S3')
dtype([('f0', '<i8'), ('f1', '<f4'), ('f2', 'S3')])

np.dtype('3int8, float32, (2, 3)float64')
dtype([('f0', 'i1', (3,)), ('f1', '<f4'), ('f2', '<f8', (2, 3))])



np.dtype({'names': ['col1', 'col2'], 'formats': ['i4', 'f4']})
dtype([('col1', '<i4'), ('col2', '<f4')])

np.dtype({'names': ['col1', 'col2'],

          'formats': ['i4', 'f4'],

          'offsets': [0, 4],

          'itemsize': 12})
dtype({'names': ['col1', 'col2'], 'formats': ['<i4', '<f4'], 'offsets': [0, 4], 'itemsize': 12})





np.dtype({'col1': ('i1', 0), 'col2': ('f4', 1)})
dtype([('col1', 'i1'), ('col2', '<f4')])





d = np.dtype([('x', 'i8'), ('y', 'f4')])

d.names
('x', 'y')




d['x']
dtype('int64')




d.fields
mappingproxy({'x': (dtype('int64'), 0), 'y': (dtype('float32'), 8)})




def print_offsets(d):

    print("offsets:", [d.fields[name][1] for name in d.names])

    print("itemsize:", d.itemsize)

print_offsets(np.dtype('u1, u1, i4, u1, i8, u2'))
offsets: [0, 1, 2, 6, 7, 15]
itemsize: 17




print_offsets(np.dtype('u1, u1, i4, u1, i8, u2', align=True))
offsets: [0, 1, 4, 8, 16, 24]
itemsize: 32



np.dtype([(('my title', 'name'), 'f4')])
dtype([(('my title', 'name'), '<f4')])



np.dtype({'name': ('i4', 0, 'my title')})
dtype([(('my title', 'name'), '<i4')])



for name in d.names:

    print(d.fields[name][:2])
(dtype('int64'), 0)
(dtype('float32'), 8)




x = np.array([(1, 2, 3), (4, 5, 6)], dtype='i8, f4, f8')

x[1] = (7, 8, 9)

x
array([(1, 2., 3.), (7, 8., 9.)],
     dtype=[('f0', '<i8'), ('f1', '<f4'), ('f2', '<f8')])




x = np.zeros(2, dtype='i8, f4, ?, S1')

x[:] = 3

x
array([(3, 3., True, b'3'), (3, 3., True, b'3')],
      dtype=[('f0', '<i8'), ('f1', '<f4'), ('f2', '?'), ('f3', 'S1')])

x[:] = np.arange(2)

x
array([(0, 0., False, b'0'), (1, 1., True, b'1')],
      dtype=[('f0', '<i8'), ('f1', '<f4'), ('f2', '?'), ('f3', 'S1')])





twofield = np.zeros(2, dtype=[('A', 'i4'), ('B', 'i4')])

onefield = np.zeros(2, dtype=[('A', 'i4')])

nostruct = np.zeros(2, dtype='i4')

nostruct[:] = twofield
Traceback (most recent call last):
...
TypeError: Cannot cast array data from dtype([('A', '<i4'), ('B', '<i4')]) to dtype('int32') according to the rule 'unsafe'












a = np.zeros(3, dtype=[('a', 'i8'), ('b', 'f4'), ('c', 'S3')])

b = np.ones(3, dtype=[('x', 'f4'), ('y', 'S3'), ('z', 'O')])

b[:] = a

b
array([(0., b'0.0', b''), (0., b'0.0', b''), (0., b'0.0', b'')],
      dtype=[('x', '<f4'), ('y', 'S3'), ('z', 'O')])








x = np.array([(1, 2), (3, 4)], dtype=[('foo', 'i8'), ('bar', 'f4')])

x['foo']
array([1, 3])

x['foo'] = 10

x
array([(10, 2.), (10, 4.)],
      dtype=[('foo', '<i8'), ('bar', '<f4')])






y = x['bar']

y[:] = 11

x
array([(10, 11.), (10, 11.)],
      dtype=[('foo', '<i8'), ('bar', '<f4')])




y.dtype, y.shape, y.strides
(dtype('float32'), (2,), (12,))





x = np.zeros((2, 2), dtype=[('a', np.int32), ('b', np.float64, (3, 3))])

x['a'].shape
(2, 2)

x['b'].shape
(2, 2, 3, 3)






a = np.zeros(3, dtype=[('a', 'i4'), ('b', 'i4'), ('c', 'f4')])

a[['a', 'c']]
array([(0, 0.), (0, 0.), (0, 0.)],
     dtype={'names': ['a', 'c'], 'formats': ['<i4', '<f4'], 'offsets': [0, 8], 'itemsize': 12})





a[['a', 'c']].view('i8')  # Fails in Numpy 1.16
Traceback (most recent call last):
   File "<stdin>", line 1, in <module>
ValueError: When changing to a smaller dtype, its size must be a divisor of the size of original dtype





from numpy.lib.recfunctions import repack_fields

repack_fields(a[['a', 'c']]).view('i8')  # supported in 1.16
array([0, 0, 0])





b = np.zeros(3, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])

b[['x', 'z']].view('f4')
array([0., 0., 0., 0., 0., 0., 0., 0., 0.], dtype=float32)





from numpy.lib.recfunctions import structured_to_unstructured

structured_to_unstructured(b[['x', 'z']])
array([[0., 0.],
       [0., 0.],
       [0., 0.]], dtype=float32)






a[['a', 'c']] = (2, 3)

a
array([(2, 0, 3.), (2, 0, 3.), (2, 0, 3.)],
      dtype=[('a', '<i4'), ('b', '<i4'), ('c', '<f4')])





a[['a', 'c']] = a[['c', 'a']]




x = np.array([(1, 2., 3.)], dtype='i, f, f')

scalar = x[0]

scalar
(1, 2., 3.)

type(scalar)
<class 'numpy.void'>




x = np.array([(1, 2), (3, 4)], dtype=[('foo', 'i8'), ('bar', 'f4')])

s = x[0]

s['bar'] = 100

x
array([(1, 100.), (3, 4.)],
      dtype=[('foo', '<i8'), ('bar', '<f4')])





scalar = np.array([(1, 2., 3.)], dtype='i, f, f')[0]

scalar[0]
1

scalar[1] = 4





scalar.item(), type(scalar.item())
((1, 4.0, 3.0), <class 'tuple'>)





a = np.array([(1, 1), (2, 2)], dtype=[('a', 'i4'), ('b', 'i4')])

b = np.array([(1, 1), (2, 3)], dtype=[('a', 'i4'), ('b', 'i4')])

a == b
array([True, False])





b = np.array([(1.0, 1), (2.5, 2)], dtype=[("a", "f4"), ("b", "i4")])

a == b
array([True, False])






np.result_type(np.dtype("i,>i"))
dtype([('f0', '<i4'), ('f1', '<i4')])

np.result_type(np.dtype("i,>i"), np.dtype("i,i"))
dtype([('f0', '<i4'), ('f1', '<i4')])




dt = np.dtype("i1,V3,i4,V1")[["f0", "f2"]]

dt
dtype({'names':['f0','f2'], 'formats':['i1','<i4'], 'offsets':[0,4], 'itemsize':9})

np.result_type(dt)
dtype([('f0', 'i1'), ('f2', '<i4')])





dt = np.dtype("i1,V3,i4,V1", align=True)[["f0", "f2"]]

dt
dtype({'names':['f0','f2'], 'formats':['i1','<i4'], 'offsets':[0,4], 'itemsize':12}, align=True)

np.result_type(dt)
dtype([('f0', 'i1'), ('f2', '<i4')], align=True)

np.result_type(dt).isalignedstruct
True





np.result_type(np.dtype("i,i"), np.dtype("i,i", align=True))
dtype([('f0', '<i4'), ('f1', '<i4')], align=True)




recordarr = np.rec.array([(1, 2., 'Hello'), (2, 3., "World")],

                   dtype=[('foo', 'i4'),('bar', 'f4'), ('baz', 'S10')])

recordarr.bar
array([2., 3.], dtype=float32)

recordarr[1:2]
rec.array([(2, 3., b'World')],
      dtype=[('foo', '<i4'), ('bar', '<f4'), ('baz', 'S10')])

recordarr[1:2].foo
array([2], dtype=int32)

recordarr.foo[1:2]
array([2], dtype=int32)

recordarr[1].baz
b'World'




arr = np.array([(1, 2., 'Hello'), (2, 3., "World")],

            dtype=[('foo', 'i4'), ('bar', 'f4'), ('baz', 'S10')])

recordarr = np.rec.array(arr)




arr = np.array([(1, 2., 'Hello'), (2, 3., "World")],

               dtype=[('foo', 'i4'),('bar', 'f4'), ('baz', 'a10')])

recordarr = arr.view(dtype=np.dtype((np.record, arr.dtype)),

                     type=np.recarray)



recordarr = arr.view(np.recarray)

recordarr.dtype
dtype((numpy.record, [('foo', '<i4'), ('bar', '<f4'), ('baz', 'S10')]))





arr2 = recordarr.view(recordarr.dtype.fields or recordarr.dtype, np.ndarray)



recordarr = np.rec.array([('Hello', (1, 2)), ("World", (3, 4))],

                dtype=[('foo', 'S6'),('bar', [('A', int), ('B', int)])])

type(recordarr.foo)
<class 'numpy.ndarray'>

type(recordarr.bar)
<class 'numpy.recarray'>




numpy.lib.recfunctions.append_fields(base, names, data, dtypes=None, fill_value=-1, usemask=True, asrecarray=False)




from numpy.lib import recfunctions as rfn

b = np.array([(1, 2, 5), (4, 5, 7), (7, 8 ,11), (10, 11, 12)],

             dtype=[('x', 'i4'), ('y', 'f4'), ('z', 'f8')])

rfn.apply_along_fields(np.mean, b)
array([ 2.66666667,  5.33333333,  8.66666667, 11.        ])

rfn.apply_along_fields(np.mean, b[['x', 'z']])
array([ 3. ,  5.5,  9. , 11. ])




from numpy.lib import recfunctions as rfn

a = np.array([(1, (2, 3.0)), (4, (5, 6.0))],

  dtype=[('a', np.int64), ('b', [('ba', np.double), ('bb', np.int64)])])

rfn.drop_fields(a, 'a')
array([((2., 3),), ((5., 6),)],
      dtype=[('b', [('ba', '<f8'), ('bb', '<i8')])])

rfn.drop_fields(a, 'ba')
array([(1, (3,)), (4, (6,))], dtype=[('a', '<i8'), ('b', [('bb', '<i8')])])

rfn.drop_fields(a, ['ba', 'bb'])
array([(1,), (4,)], dtype=[('a', '<i8')])







from numpy.lib import recfunctions as rfn

ndtype = [('a', int)]

a = np.ma.array([1, 1, 1, 2, 2, 3, 3],

        mask=[0, 0, 1, 0, 0, 0, 1]).view(ndtype)

rfn.find_duplicates(a, ignoremask=True, return_index=True)
(masked_array(data=[(1,), (1,), (2,), (2,)],
             mask=[(False,), (False,), (False,), (False,)],
       fill_value=(999999,),
            dtype=[('a', '<i8')]), array([0, 1, 3, 4]))




from numpy.lib import recfunctions as rfn

ndtype = np.dtype([('a', '<i4'), ('b', [('ba', '<f8'), ('bb', '<i4')])])

rfn.flatten_descr(ndtype)
(('a', dtype('int32')), ('ba', dtype('float64')), ('bb', dtype('int32')))




from numpy.lib import recfunctions as rfn

ndtype =  np.dtype([('A', int),

                    ('B', [('BA', int),

                           ('BB', [('BBA', int), ('BBB', int)])])])

rfn.get_fieldstructure(ndtype)

# XXX: possible regression, order of BBA and BBB is swapped
{'A': [], 'B': [], 'BA': ['B'], 'BB': ['B'], 'BBA': ['B', 'BB'], 'BBB': ['B', 'BB']}






from numpy.lib import recfunctions as rfn

rfn.get_names(np.empty((1,), dtype=[('A', int)]).dtype)
('A',)

rfn.get_names(np.empty((1,), dtype=[('A',int), ('B', float)]).dtype)
('A', 'B')

adtype = np.dtype([('a', int), ('b', [('ba', int), ('bb', int)])])

rfn.get_names(adtype)
('a', ('b', ('ba', 'bb')))






from numpy.lib import recfunctions as rfn

rfn.get_names_flat(np.empty((1,), dtype=[('A', int)]).dtype) is None
False

rfn.get_names_flat(np.empty((1,), dtype=[('A',int), ('B', str)]).dtype)
('A', 'B')

adtype = np.dtype([('a', int), ('b', [('ba', int), ('bb', int)])])

rfn.get_names_flat(adtype)
('a', 'b', 'ba', 'bb')






from numpy.lib import recfunctions as rfn

rfn.merge_arrays((np.array([1, 2]), np.array([10., 20., 30.])))
array([( 1, 10.), ( 2, 20.), (-1, 30.)],
      dtype=[('f0', '<i8'), ('f1', '<f8')])

rfn.merge_arrays((np.array([1, 2], dtype=np.int64),

        np.array([10., 20., 30.])), usemask=False)
 array([(1, 10.0), (2, 20.0), (-1, 30.0)],
         dtype=[('f0', '<i8'), ('f1', '<f8')])

rfn.merge_arrays((np.array([1, 2]).view([('a', np.int64)]),

              np.array([10., 20., 30.])),

             usemask=False, asrecarray=True)
rec.array([( 1, 10.), ( 2, 20.), (-1, 30.)],
          dtype=[('a', '<i8'), ('f1', '<f8')])






from numpy.lib import recfunctions as rfn

a = np.array([(1, 10.), (2, 20.)], dtype=[('A', np.int64), ('B', np.float64)])

b = np.zeros((3,), dtype=a.dtype)

rfn.recursive_fill_fields(a, b)
array([(1, 10.), (2, 20.), (0,  0.)], dtype=[('A', '<i8'), ('B', '<f8')])




from numpy.lib import recfunctions as rfn

a = np.array([(1, (2, [3.0, 30.])), (4, (5, [6.0, 60.]))],

  dtype=[('a', int),('b', [('ba', float), ('bb', (float, 2))])])

rfn.rename_fields(a, {'a':'A', 'bb':'BB'})
array([(1, (2., [ 3., 30.])), (4, (5., [ 6., 60.]))],
      dtype=[('A', '<i8'), ('b', [('ba', '<f8'), ('BB', '<f8', (2,))])])




from numpy.lib import recfunctions as rfn

def print_offsets(d):

    print("offsets:", [d.fields[name][1] for name in d.names])

    print("itemsize:", d.itemsize)


dt = np.dtype('u1, <i8, <f8', align=True)

dt
dtype({'names': ['f0', 'f1', 'f2'], 'formats': ['u1', '<i8', '<f8'], 'offsets': [0, 8, 16], 'itemsize': 24}, align=True)

print_offsets(dt)
offsets: [0, 8, 16]
itemsize: 24

packed_dt = rfn.repack_fields(dt)

packed_dt
dtype([('f0', 'u1'), ('f1', '<i8'), ('f2', '<f8')])

print_offsets(packed_dt)
offsets: [0, 1, 9]
itemsize: 17





from numpy.lib import recfunctions as rfn

a = np.ones(4, dtype=[('a', 'i4'), ('b', 'f8'), ('c', 'u1')])

rfn.require_fields(a, [('b', 'f4'), ('c', 'u1')])
array([(1., 1), (1., 1), (1., 1), (1., 1)],
  dtype=[('b', '<f4'), ('c', 'u1')])

rfn.require_fields(a, [('b', 'f4'), ('newf', 'u1')])
array([(1., 0), (1., 0), (1., 0), (1., 0)],
  dtype=[('b', '<f4'), ('newf', 'u1')])




from numpy.lib import recfunctions as rfn

x = np.array([1, 2,])

rfn.stack_arrays(x) is x
True

z = np.array([('A', 1), ('B', 2)], dtype=[('A', '|S3'), ('B', float)])

zz = np.array([('a', 10., 100.), ('b', 20., 200.), ('c', 30., 300.)],

  dtype=[('A', '|S3'), ('B', np.double), ('C', np.double)])

test = rfn.stack_arrays((z,zz))

test
masked_array(data=[(b'A', 1.0, --), (b'B', 2.0, --), (b'a', 10.0, 100.0),
                   (b'b', 20.0, 200.0), (b'c', 30.0, 300.0)],
             mask=[(False, False,  True), (False, False,  True),
                   (False, False, False), (False, False, False),
                   (False, False, False)],
       fill_value=(b'N/A', 1.e+20, 1.e+20),
            dtype=[('A', 'S3'), ('B', '<f8'), ('C', '<f8')])




from numpy.lib import recfunctions as rfn

a = np.zeros(4, dtype=[('a', 'i4'), ('b', 'f4,u2'), ('c', 'f4', 2)])

a
array([(0, (0., 0), [0., 0.]), (0, (0., 0), [0., 0.]),
       (0, (0., 0), [0., 0.]), (0, (0., 0), [0., 0.])],
      dtype=[('a', '<i4'), ('b', [('f0', '<f4'), ('f1', '<u2')]), ('c', '<f4', (2,))])

rfn.structured_to_unstructured(a)
array([[0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0.]])

b = np.array([(1, 2, 5), (4, 5, 7), (7, 8 ,11), (10, 11, 12)],

             dtype=[('x', 'i4'), ('y', 'f4'), ('z', 'f8')])

np.mean(rfn.structured_to_unstructured(b[['x', 'z']]), axis=-1)
array([ 3. ,  5.5,  9. , 11. ])







from numpy.lib import recfunctions as rfn

dt = np.dtype([('a', 'i4'), ('b', 'f4,u2'), ('c', 'f4', 2)])

a = np.arange(20).reshape((4,5))

a
array([[ 0,  1,  2,  3,  4],
       [ 5,  6,  7,  8,  9],
       [10, 11, 12, 13, 14],
       [15, 16, 17, 18, 19]])

rfn.unstructured_to_structured(a, dt)
array([( 0, ( 1.,  2), [ 3.,  4.]), ( 5, ( 6.,  7), [ 8.,  9.]),
       (10, (11., 12), [13., 14.]), (15, (16., 17), [18., 19.])],
      dtype=[('a', '<i4'), ('b', [('f0', '<f4'), ('f1', '<u2')]), ('c', '<f4', (2,))])








