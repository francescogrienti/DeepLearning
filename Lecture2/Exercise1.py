import numpy as np 

a = np.array([[0.5, -1],[-1, 2]], dtype=np.float32)
shape = a.shape
dim = a.ndim
print(shape, dim)
b = np.copy(a.flatten())
b[0::2] = 0 #this syntax gets every even index in a list
print(b)



