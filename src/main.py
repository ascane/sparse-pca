from math import floor, sqrt
import numpy as np

import data_generator, sdp_estimator

def loss(u, v):
    return sqrt(1 - (u.dot(v)) ** 2)

def mean_loss_lin(p, mu, n_iter=1, verbose=False):
    k = int(floor(sqrt(p)))
    n = int(floor(mu * k * np.log10(p)))
    return _mean_loss(p, n, k, n_iter, verbose)

def mean_loss_quad(p, mu, n_iter=1, verbose=False):
    k = int(floor(sqrt(p)))
    n = int(floor(mu * k * k * np.log10(p)))
    return _mean_loss(p, n, k, n_iter, verbose)

def _mean_loss(p, n, k, n_iter, verbose):
    if verbose:
        print "p = %s, n = %s, k = %s" %(p, n, k)
    dg = data_generator.DataGenerator(p, n, k)
    loss_sum = 0
    for i in range(n_iter):
        print "round %d" %(i + 1)
        dg.reset()
        X = dg.get_X()
        v = dg.get_v()
        sdp = sdp_estimator.SdpEstimator(X, lbda=0.1, eps=0.1)
        v_sdp = sdp.get_sdp()
        l = loss(v, v_sdp)
        loss_sum += l
        if verbose:
            print "X ="
            print X
            print "v ="
            print v
            print "v_sdp ="
            print v_sdp
            print "loss = %f" %l
    return loss_sum / n_iter

if __name__ == "__main__":
    mean_l = mean_loss_lin(p=50, mu=500, n_iter=2, verbose=True)
    print "mean loss = %f" %mean_l