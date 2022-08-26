import numpy as np
class polynomialfeature:
    def __init__(self,number_of_basis):
        self.n=number_of_basis
        
    def fit(self,x):
        X=[]
        for i in range(self.n):
            X.append(x**(i+1))
        return np.array(X).T
        

