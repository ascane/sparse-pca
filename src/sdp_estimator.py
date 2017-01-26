from math import ceil, sqrt
import numpy as np

class SdpEstimator(object):
    def __init__(self, X, lbda, eps):
        assert lbda > 0
        assert eps > 0
        self.X = X
        self.lbda = lbda
        self.eps = eps
        self.n = self.X.shape[0]
        self.p = self.X.shape[1]
        self.sigma_hat = None
        self.M_hat_eps = None
        self.sdp_hat = None
    
    def get_sdp(self):
        if self.sdp_hat is None:
            self._sdp()
        return self.sdp_hat
        
    def _sdp(self):
        if self.sigma_hat is None:
            self._sigma_hat()
        if self.M_hat_eps is None:
            self._M_hat_eps()
        if self.sdp_hat is None:
            self._sdp_hat()
    
    def _sigma_hat(self):
        self.sigma_hat = self.X.transpose().dot(self.X) / self.n
    
    def _M_hat_eps(self):
        M = np.identity(self.p) / self.p
        U = np.zeros((self.p, self.p))
        N = ceil((self.lbda * self.lbda * self.p * self.p + 1)/ sqrt(2) / self.eps)
        self.M_hat_eps = np.zeros((self.p, self.p))
        for _ in range(N):
            U2 = self._proj_sym_U(U - 1 / sqrt(2) * M)
            M2 = self._proj_sym_M(M + 1 / sqrt(2) * self.sigma_hat + 1 / sqrt(2) * U)
            self.M_hat_eps += M2
            U = self._proj_sym_U(U - 1 / sqrt(2) * M2)
            M = self._proj_sym_M(M + 1 / sqrt(2) * self.sigma_hat + 1 / sqrt(2) * U2)
        self.M_hat_eps /= N
            
    def _proj_sym_U(self, A):
        A2 = A.clone()
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                if self.lbda < np.abs(A[i, j]):
                    A2[i, j] = np.sign(A[i, j]) * self.lbda 
        return A2
    
    def _proj_sym_M(self, A):
        # decompose A = P D P^T for some orthogonal P and diagonal D = diag(d)
        P, d, Q = np.linalg.svd(A)
        for i in range(A.shape[0]):
            if np.sign(P[0, i]) != np.sign(Q[i, 0]):
                d[i] *= -1
        # project d on the unit (p - 1)-simplex W:={(w_1, w_2, ..., w_p): w_i >= 0, \sum_j w_j = 1}
        s = np.sum(d)
        if np.abs(s - 1) > 1e-5:
            d +=  (1 - s) / self.p
        # transform A back
        return P.dot(np.diag(d).dot(P.transpose()))
    
    def _sdp_hat(self):
        # TODO
        pass
