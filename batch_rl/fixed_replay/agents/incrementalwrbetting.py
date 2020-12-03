class IncrementalWRBetting:
    def __init__(self, decay, taumax=0.25):
        from math import floor, log
        assert 0 < decay <= 1
        self.decay = decay
        self.n = 0
        self.sumwr = 0
        self.sumwrsq = 0

        assert 0 <= taumax < 1
        self.taumax = taumax

    def tfhook(self, gamma, w, r):
        import numpy as np

        q = []
        v = self.findv()
        tau = self.bet(v)
        for gn, wn, rn in zip(gamma, w, r):
            wr = (gn * wn).dot(rn) / max(1, len(wn))
            q.append(1.0 / (1.0 + tau * (wr - v)))
            self.observe(tau, wr)

        return np.array(q).astype(np.single)

    def observe(self, tau, wr):
        from math import log1p

        self.n *= self.decay
        self.n += 1
        self.sumwr *= self.decay
        self.sumwr += wr
        self.sumwrsq *= self.decay
        self.sumwrsq += wr**2

    def bet(self, v):
        mean = self.sumwr - self.n * v
        assert mean >= 0, mean
        var = self.sumwrsq - 2 * v * self.sumwr + self.n * v**2
        assert var >= 0, var

        tauub = min(self.taumax, 0.5 / v if v > 0.5 else 1)

        return min(tauub, max(0, mean / (mean + var) if var > 0 else 1 if mean >= 0 else 0))

    def findv(self):
        from math import sqrt

        meanwr = self.sumwr / max(1, self.n)
        varwr = sqrt(max(0, self.sumwrsq / max(1, self.n) - meanwr**2))
        vadj = max(0, meanwr - 3 * varwr / sqrt(max(1, self.n)))
        return vadj

