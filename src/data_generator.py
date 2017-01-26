import numpy as np
from math import sqrt

class DataGenerator(object):
    def __init__(self, p, n, k, theta=1.):
        self.p = p
        self.n = n
        self.k = k
        self.theta = theta
        self.v = None
        self.X = None
        
    def get_X(self):
        if self.X is None:
            self._generate()
        return self.X
    
    def get_v(self):
        if self.v is None:
            self._generate()
        return self.v
    
    def _generate(self):
        self.v = np.zeros(self.p)
        for i in range(self.k):
            self.v[i] = 1 / sqrt(self.k)
        sigma = np.identity(self.p) + self.theta * self.v[:, np.newaxis] * self.v[np.newaxis, :]
        self.X = np.random.multivariate_normal(np.zeros(self.p), sigma, self.n)

if __name__ == "__main__":
    dg = DataGenerator(2, 5, 2)
    print dg.get_X()
