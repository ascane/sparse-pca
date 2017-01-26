from math import floor, sqrt

import data_generator, sdp_estimator

def loss(u, v):
    return sqrt(1 - (u.dot(v)) ** 2)

if __name__ == "__main__":
    p = 50
    n = 334
    k = int(floor(sqrt(p)))
    dg = data_generator.DataGenerator(p, n, k)
    X = dg.get_X()
    v = dg.get_v()
    print "X = "
    print X
    print "v = "
    print v

    sdp = sdp_estimator.SdpEstimator(X, lbda=0.1, eps=0.1)
    v_sdp = sdp.get_sdp()
    print "v_sdp = "
    print v_sdp
    
    print "loss = "
    print loss(v, v_sdp)
