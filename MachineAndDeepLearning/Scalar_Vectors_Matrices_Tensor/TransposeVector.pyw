#Import Relevant Libraries

import numpy as np

#creating matrices

m1 = np.array([[5,12,6],[-3,0,14]])
m1

m2 = np.array([[9,8,7],[1,3,-5]])
m2

#Transpose matrices

m1.T
m2.T
#2 new matrices for difference

m3 = np.array([[5,3],[-2,4]])
m4 = np.array([[7,-5],[3,8]])
m3 - m4

#Adding Vectors Together

v1 = np.array([1,2,3,4,5])

#Transpose of vector will look same because its 1D,Inorder to see the effect of Transpose we need to convert Vector into 2D
v1.T
#Reshaping vector to 2D
v1_Shape = v1.reshape(1,3)

#Now we can spot actual change in vector
v1_Shape.T
