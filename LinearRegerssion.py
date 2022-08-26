import numpy as np
import polynomialfeature as pr
class LinearRegerssion:
    def __init__(self,learning_rate=1e-5,lambda_=0,tolerance=0.5, degree = 1, maxiter=10000):
        self.l=learning_rate
        self.lambda_=lambda_ 
        self.w=None
        self.tol=tolerance
        self.maxitr = maxiter
        self.n_basis = degree

    def fit(self,X,y):
        clf1=pr.polynomialfeature(self.n_basis)
        X = clf1.fit(X)
        m,n=X.shape
        y=y.reshape(m,1)
        X = np.hstack([np.ones(m).reshape(m,1), X])
        m,n=X.shape
        self.w = np.ones(n).reshape(1,n)
        
        mse = []
        j=0
        while True:
            cost = np.sum(np.square((X@self.w.T-y)))/(2*m)+self.lambda_ /2*float(np.dot(self.w,self.w.T))
            dw= (X.T@(X@self.w.T-y)).T
            self.w -= self.l*dw+self.lambda_ *self.w
            mse.append(cost)
            if cost <=self.tol:
                break
            elif j == self.maxitr:
                [print("model does not fit "+str(j))]
                break
            elif type(j%1000)==int:
                print(cost)
            j=j+1
        return np.array(mse)
    
    def predict(self,x):
        clf1=pr.polynomialfeature(self.n_basis)
        x = clf1.fit(x)
        m,n=x.shape
        x = np.hstack([np.ones(m).reshape(m,1), x])
        return x@self.w.T
