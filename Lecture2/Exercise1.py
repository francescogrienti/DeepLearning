import numpy as np 

#Basic review of some functionalities of Numpy

a = np.array([[0.5, -1],[-1, 2]], dtype=np.float32)
shape = a.shape
dim = a.ndim
print(shape, dim)

#Deep copy of a 
b = np.copy(a.flatten())

#Set 0 in even indeces
b[0::2] = 0 
print(b)
