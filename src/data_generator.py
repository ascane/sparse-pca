from math import sqrt
import numpy as np

class DataGenerator(object):
    """A class that generates X = (X_1, ..., X_n)^T, where X_1, ..., X_n ~(i.i.d.) N_p(0, Sigma),
    Sigma = I_p + theta v_1 v_1^T, v_1 k-sparse.
    """
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
    
    def reset(self):
        self.X = None
        self.v = None
    
    def _generate(self):
        self.v = np.zeros(self.p)
        for i in range(self.k):
            self.v[i] = 1 / sqrt(self.k)
        sigma = np.identity(self.p) + self.theta * self.v[:, np.newaxis] * self.v[np.newaxis, :]
        self.X = np.random.multivariate_normal(np.zeros(self.p), sigma, self.n)

if __name__ == "__main__":
    dg = DataGenerator(2, 5, 2)
    print dg.get_X()
