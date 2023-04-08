import numpy as np
from numba import njit
import random 
import time


#Implement a function that performs the dot product using only python primitives
@njit
def old_dot_product(a, b):
    if len(b) != len(a[0]): 
        raise ValueError()
    else: 
        c = [[0 for i in range(len(b[0]))] for j in range(len(a))]
        for i in range(len(a)):
            for j in range(len(b[0])):
                for k in range(len(a[0])):
                    c[i][j] += a[i][k]*b[k][j]
        return c 


#NumPy's dot product 
def num_dot_product(a,b): 
    return np.dot(a,b)


def main(): 
    #Space dimension N 
    N = 100

    #Allocate a random vector v of length N 
    v = np.array([random.uniform(2, 50) for i in range(N)], dtype = np.float64)

    #Allocate a random square matrix M of dimension (NxN)
    A = np.array([[random.uniform(1, 3) for i in range(N)] for j in range(N)])

    #Performance test on dot product using python primitives
    st_1= time.time()
    C = old_dot_product(A,A)
    et_1 = time.time()
    exec_time_1 = et_1-st_1
    print("Execution time dot product without NumPy:", exec_time_1, "seconds")

    #Performance test on dot product using NumPy's dot product
    st_2= time.time()
    D = num_dot_product(A,A)
    et_2 = time.time()
    exec_time_2 = et_2-st_2
    print("Execution time dot product with NumPy:", exec_time_2, "seconds")



if __name__ == "__main__":
    main()


    