"""
Construction and evaluation of a 1-D polynomial function
"""

class oneDimPol():
    def __init__(self, degree:int) -> None:
        self.degree = degree 
        self.param = [0 for i in range(self.degree+1)]
    
    def set_parameters(self, param:list)-> None:
        if len(param) != len(self.param):
            raise ValueError("Wrong number of parameters")
        else: 
            self.param = param 
        
    def get_parameters(self) -> list: 
        return self.param
    
    def execute(self, x:float) -> float: 
        eval = 0.
        for i in range(len(self.param)): 
            eval += self.param[i]*pow(x, self.degree-i)

        return eval 

    def call(self, x) -> float: 
        return self.execute(x)
    

def main():
    OneDimension = oneDimPol(2)
    OneDimension.set_parameters([1., 2., 4.])
    print(OneDimension.get_parameters())
    print(OneDimension.call(2.0))


if __name__ == "__main__":
    main()


        