class IncrementalIwLb:
    def __init__(self, coverage, nbatches):
        from collections import deque
        self.coverage = coverage
        self.batches = deque(maxlen=nbatches)
        self.vhat = 0
        self.alphastar = 1
        self.kappastar = 1
        
    def dualstfhook(self):
        import numpy as np
        
        return np.array([ self.vhat, self.alphastar, 0, self.kappastar ]).astype(np.single)
    
    def tfhook(self, g, w, r):
        from scipy.stats import f
        from scipy import optimize
        import numpy as np
        
        self.batches.append( [ wn.dot(gn * np.clip(rn, a_min=0, a_max=None)) for gn, wn, rn in zip(g, w, r) ] )

        N = len(self.batches) * g.shape[0]
        Delta = 0.5 * f.isf(q=1.0-self.coverage, dfn=1, dfd=N-1) / N
        phi = -Delta
        
        def kappa(alpha):
            from math import fsum, log, exp
            return exp(phi + (1/N) * fsum(log(alpha + wdotr) for v in self.batches for wdotr in v))

        def dual(alpha):
            return alpha - kappa(alpha)
        
        alphamin = 1e-6
        alphamax = N
        res = optimize.minimize_scalar(fun=dual, method='bounded', bounds=(alphamin, alphamax))
        self.vhat = -res.fun
        self.alphastar = res.x
        self.kappastar = kappa(self.alphastar)
        
        return np.array([ self.kappastar / (self.alphastar + wn.dot(gn * np.clip(rn, a_min=0, a_max=None))) for gn, wn, rn in zip(g, w, r) ]).astype(np.single)
