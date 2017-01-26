import numpy as np
from math import sqrt

class DataGenerator(object):
    def __init__(self, p, n, k, theta=1.):
        self.p = p
        self.n = n
        self.k = k
        self.theta = theta
        self.X = None
        
    def getX(self):
        if self.X is None:
            self._generate()
        return self.X
    
    def _generate(self):
        v = np.zeros(self.p)
        for i in range(self.k):
            v[i] = 1 / sqrt(self.k)
        sigma = np.identity(self.p) + self.theta * v[:, np.newaxis] * v[np.newaxis, :]
        self.X = np.random.multivariate_normal(np.zeros(self.p), sigma, self.n)

if __name__ == "__main__":
    dg = DataGenerator(2, 5, 2)
    print dg.getX()
    