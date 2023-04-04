import numpy as np
from abc import ABC, abstractclassmethod

class Variable:
    def __init__(self, name):
        self.name = name

    @abstractclassmethod
    def sample(self, size):
        raise NotImplementedError()


class Normal(Variable): 
    def __init__(self, name): 
        super().__init__(name)
        self.mu = 0
        self.sigma = 1 

    def sample(self, size):
        s = np.random.normal(self.mu, self.sigma, size)
        return s


def main(): 
    Gauss = Normal("FirstClassDL")
    n = Gauss.sample(100)
    print(n)
 

if __name__== "__main__": 
    main()

