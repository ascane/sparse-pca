from math import floor, sqrt

import data_generator, sdp_estimator

if __name__ == "__main__":
    p = 50
    n = 50
    k = int(floor(sqrt(p)))
    dg = data_generator.DataGenerator(p, n, k)
    X = dg.getX()
    print X

    sdp = sdp_estimator.SdpEstimator(X, lbda=0.1, eps=0.1)
    print sdp.get_sdp()
