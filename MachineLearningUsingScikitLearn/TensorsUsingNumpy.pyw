#import the numpy package
import numpy as np

#Creating 2 Marices
M1=np.array([[5,12,6],[-3,0,14]])

M2=np.array([[9,8,7],[1,3,-5]])

#Creating tensors nothing just merging 2 matrices in one

Tensor=np.array([M1,M2])

#Print Tensor
Tensor
#Checking Shape of Tensor
Tensor.shape
