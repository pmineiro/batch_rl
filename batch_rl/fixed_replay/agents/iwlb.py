class IwLb:
    def __init__(self, coverage=0.9):
        self.coverage = coverage
        self.alphastar = 1
        self.kappastar = 1
        self.vhat = 0

    def dualstfhook(self):
        import numpy as np

        return np.array([ self.vhat, self.alphastar, self.kappastar ]).astype(np.single)

    def tfhook(self, g, w, r):
        from scipy.stats import f
        from scipy import optimize
        import numpy as np

        N = g.shape[0]
        Delta = 0.5 * f.isf(q=1.0-self.coverage, dfn=1, dfd=N-1) / N
        phi = -Delta

        def kappa(alpha):
            from math import fsum, log, exp
            return exp(phi + fsum((1/N) * log(alpha + wn.dot(gn * rn)) for gn, wn, rn in zip(g, w, r)))

        def dual(alpha):
            return alpha - kappa(alpha)

        alphamin = 1e-6
        alphamax = N
        res = optimize.minimize_scalar(fun=dual, method='bounded', bounds=(alphamin, alphamax))
        self.alphastar = res.x
        self.kappastar = kappa(self.alphastar)
        self.vhat = -res.fun

        return np.array([ self.kappastar / (self.alphastar + wn.dot(gn * rn)) for gn, wn, rn in zip(g, w, r) ]).astype(np.single)
