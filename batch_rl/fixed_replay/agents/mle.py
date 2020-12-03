class MLE:
  def __init__(self):
      self.wmax = 2

  def tfhook(self, gamma, w, r):
      import numpy as np
      from scipy import optimize

      self.wmax = max(np.max(w), self.wmax)

      wmin = 0
      wmax = self.wmax

      wdataset = lambda: (wtraj[-1] for wtraj in w)
      N = sum(1 for _ in wdataset())
      def dual(beta):
          from math import fsum
          return -fsum((1/N) * np.log(1 + beta * (w - 1)) for w in wdataset())
      betamin = (1e-5 - 1) / (wmax - 1)
      betamax = (1 - 1e-5) / (1 - wmin)
      res = optimize.minimize_scalar(fun=dual, method='bounded', bounds=(betamin, betamax))
      betamle = res.x

      return np.array([1 / (1 + betamle * (w - 1)) for w in wdataset()]).astype(np.single)
