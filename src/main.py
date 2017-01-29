from math import floor, sqrt
import numpy as np

import data_generator, sdp_estimator

def loss(u, v):
    return sqrt(1 - (u.dot(v)) ** 2)

def mean_loss_lin(p, nu, n_iter=1, verbose=False, output=None):
    k = int(floor(sqrt(p)))
    n = int(floor(nu * k * np.log10(p)))
    return _mean_loss(p, n, k, n_iter, verbose, output)

def mean_loss_quad(p, nu, n_iter=1, verbose=False, output=None):
    k = int(floor(sqrt(p)))
    n = int(floor(nu * k * k * np.log10(p)))
    return _mean_loss(p, n, k, n_iter, verbose, output)

def _mean_loss(p, n, k, n_iter, verbose, output):
    if not(output is None):
        output.write("p = %s, n = %s, k = %s \n" %(p, n, k))
    print "p = %s, n = %s, k = %s" %(p, n, k)
    dg = data_generator.DataGenerator(p, n, k)
    loss_sum = 0
    for i in range(n_iter):
        if not(output is None):
            output.write("round %d \n" %(i + 1))
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
        if not(output is None):
            output.write("loss = %f \n" %l)
        print "loss = %f" %l
    return loss_sum / n_iter

if __name__ == "__main__":
    _p = 50
    with open("../output/p=" + str(_p) + ".txt", "w") as f: 
        for _nu in range(50, 1001, 50):
            mean_l = mean_loss_lin(p=_p, nu=_nu, n_iter=100, verbose=False, output=f)
            f.write("mean loss = %f \n" %mean_l)
            print "mean loss = %f" %mean_l
