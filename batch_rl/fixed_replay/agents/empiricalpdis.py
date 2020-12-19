class EmpiricalPdis:
    def __init__(self):
        self.vhat = 0

    def dualstfhook(self):
        import numpy as np

        return np.array([ self.vhat, 1, 0, 1 ]).astype(np.single)

    def tfhook(self, g, w, r):
        import numpy as np

        self.vhat = sum(wn.dot(gn * rn) for gn, wn, rn in zip(g, w, r))

        return np.array([ 1.0 for _ in g ]).astype(np.single)
