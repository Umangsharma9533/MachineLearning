import numpy as np

#Scalars
s = 5
s
#Vectors
v = np.array([5,-2,4])
v

#Matrices
m = np.array([[5,12,6],[-3,0,14]])
m

#DataTypes
type(s)
type(v)
type(m)
s_array = np.array([5])
type(s_array)

#Data Shapes
m.shape
v.shape
s.shape
s_array.shape

#Creating column vector
v.reshape(1,3)
v.reshape(3,1)
m+s
m+s_array
v+s
v
v+s_array
m+v
