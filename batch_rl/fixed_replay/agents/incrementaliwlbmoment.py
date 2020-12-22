# SLSQP + betastar=0 detection
class IncrementalIwLbMoment:
    def __init__(self, coverage, nbatches):
        from collections import deque
        assert 0 <= coverage < 1
        self.coverage = coverage
        self.batches = deque(maxlen=nbatches)
        self.vhat = -1
        self.alphastar = 1
        self.betastar = 0
        self.kappastar = 1

    def dualstfhook(self):
        import numpy as np

        return np.array([ self.vhat, self.alphastar, self.betastar, self.kappastar ]).astype(np.single)

    def tfhook(self, g, w, r):
        from math import fsum
        from scipy.stats import f
        from scipy import optimize
        import numpy as np

        H = w.shape[1]
        self.batches.append( [ (np.sum(wn)/H, wn.dot(gn * np.clip(rn, a_min=0, a_max=None))) for gn, wn, rn in zip(g, w, r) ] )

        N = len(self.batches) * g.shape[0]
        Delta = 0.5 * f.isf(q=1.0-self.coverage, dfn=1, dfd=N-1) / N
        phi = -Delta

        def kappa(alpha, beta):
            from math import log, exp
            return exp(phi + (1/N) * fsum(log(alpha + beta * wdotone + wdotr) for v in self.batches for wdotone, wdotr in v))

        if fsum(wdotone for v in self.batches for wdotone, wdotr in v) <= N: # assume betastar = 0
            res = optimize.minimize_scalar(fun=lambda alpha: alpha - kappa(alpha, 0), method='bounded', bounds=(1e-6, N))
            self.alphastar, self.betastar = res.x, 0
            self.kappastar = kappa(self.alphastar, self.betastar)
            self.vhat = -res.fun
        else: # betastar >= 0
            def negmle(beta):
                from math import log
                return -(1/N) * fsum(log(1 + beta * (wdotone - 1)) for v in self.batches for wdotone, wdotr in v)

            res = optimize.minimize_scalar(fun=negmle, method='bounded', bounds=(1e-6, 1 - 1e-6))
            phi += res.fun

            def gradlogkappa(alpha, beta):
                from math import log
                return ( (1/N) * fsum(1 / (alpha + beta * wdotone + wdotr) for v in self.batches for wdotone, wdotr in v),
                         (1/N) * fsum(wdotone / (alpha + beta * wdotone + wdotr) for v in self.batches for wdotone, wdotr in v)
                       )

            def gradgradlogkappa(alpha, beta):
                from math import log
                return ( -(1/N) * fsum(1 / (alpha + beta * wdotone + wdotr)**2 for v in self.batches for wdotone, wdotr in v),
                         -(1/N) * fsum(wdotone / (alpha + beta * wdotone + wdotr)**2 for v in self.batches for wdotone, wdotr in v),
                         -(1/N) * fsum(wdotone**2 / (alpha + beta * wdotone + wdotr)**2 for v in self.batches for wdotone, wdotr in v)
                       )

            def negdual(x):
                return x[0] + x[1] - kappa(x[0], x[1])

            def jacnegdual(x):
                dakappa = kappa(x[0], x[1])
                gloga, glogb = gradlogkappa(x[0], x[1])

                return [ 1 - dakappa * gloga, 1 - dakappa * glogb ]

            res = optimize.minimize(fun=negdual,
                                    jac=jacnegdual,
                                    x0=(1.0, 0.0),
                                    method='SLSQP',
                                    bounds=[ (1e-6, None), (0.0, None) ],
                                   )
            self.alphastar, self.betastar = res.x
            self.kappastar = kappa(self.alphastar, self.betastar)
            self.vhat = -res.fun

        return np.array([ self.kappastar / (self.alphastar + self.betastar * np.sum(wn)/H + wn.dot(gn * np.clip(rn, a_min=0, a_max=None))) for gn, wn, rn in zip(g, w, r) ]).astype(np.single)
