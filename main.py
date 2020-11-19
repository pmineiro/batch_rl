import tensorflow as tf
from batch_rl.baselines.train import main
import dopamine


def pdislowerbound(wrs, wmin, wmax, coverage=0.9):
    from cvxopt import solvers, matrix
    from itertools import chain
    from math import fsum
    import numpy as np

    assert wmin == 0
    assert wmin < 1
    assert wmax > 1

    np.seterr(invalid='raise')

    N = len(wrs)
    T = len(wrs[0][0])

    def constraintvecs():
        w = np.zeros(T)
        scale = 1.0
        yield scale, w.copy()

        for t in range(T):
            w /= wmax
            scale /= wmax
            w[t] = 1
            yield scale, w.copy()

    # solve MLE
    if True:
        tiny = 1e-6
        n = fsum(1 for _ in wrs)
        x0 = np.zeros(T)

        def negdualobjective(beta):
            return -fsum((1 / n) * np.log(1 + np.dot(beta, w - 1)) for w, _ in wrs)

        def jacnegdualobjective(beta):
            jac = np.zeros(T)

            for i in range(T):
                jac[i] = -fsum((1 / n) * (w[i] - 1) / (1 + np.dot(beta, w - 1)) for w, _ in wrs)

            return jac

        def hessnegdualobjective(beta):
            hess = np.zeros((T, T))

            for i in range(T):
                for j in range(T):
                    if i <= j:
                        hess[i, j] = fsum(
                            (1 / n) * (w[i] - 1) * (w[j] - 1) / ((1 + np.dot(beta, w - 1)) ** 2) for w, _ in wrs)
                        hess[j, i] = hess[i, j]

            return hess

        def F(x=None, z=None):
            if x is None: return 0, matrix(x0)

            beta = np.reshape(np.array(x), -1)

            if any(1 + np.dot(beta, w - 1) < tiny for w, r in wrs):
                return None

            f = negdualobjective(beta)
            jf = jacnegdualobjective(beta)
            Df = matrix(jf).T
            if z is None: return f, Df
            hf = z[0] * hessnegdualobjective(beta)
            H = matrix(hf, hf.shape)
            return f, Df, H

        G = np.vstack([wvec - wscale for wscale, wvec in constraintvecs()])
        h = -np.array([wscale for wscale, _ in constraintvecs()])

        soln = solvers.cp(F, G=-matrix(G, G.shape), h=-matrix(h), options={'show_progress': False})
        from pprint import pformat
        assert soln['status'] == 'optimal', pformat(soln)
        likelihoodminusempiricalentropy = soln['primal objective']
        mlebetastar = np.reshape(np.array(soln['x']), -1)

    # solve bound
    if True:
        from scipy.stats import f

        Delta = 0.5 * f.isf(q=1.0 - coverage, dfn=1, dfd=N - 1) / N
        phiminusempiricalentropy = likelihoodminusempiricalentropy - Delta

        tiny = 1e-6
        sumwminus1 = np.array([fsum((1 / n) * (w[i] - 1) for w, r in wrs) for i in range(T)])
        sumwdotr = fsum((1 / n) * np.dot(w, r) for w, r in wrs)

        def negdualobjective(x):
            from scipy import special

            kappa, alpha, beta = x[0], x[1], x[2:]

            return (- sumwdotr
                    - np.dot(beta, sumwminus1)
                    - kappa * phiminusempiricalentropy
                    + fsum((1 / n) * special.kl_div(kappa, alpha + np.dot(beta, w - 1) + np.dot(w, r)) for w, r in wrs)
                    )

        def dkldivdx(x, y):
            # x log(x/y) - x + y
            return np.log(x / y)

        def dkldivdy(x, y):
            return 1 - x / y

        def jacnegdualobjective(x):
            kappa, alpha, beta = x[0], x[1], x[2:]

            jacobj = np.empty_like(x)

            jacobj[0] = (- phiminusempiricalentropy
                         + fsum((1 / n) * dkldivdx(kappa, alpha + np.dot(beta, w - 1) + np.dot(w, r)) for w, r in wrs)
                         )
            jacobj[1] = fsum((1 / n) * dkldivdy(kappa, alpha + np.dot(beta, w - 1) + np.dot(w, r)) for w, r in wrs)
            jacobj[2:] = -sumwminus1

            for i in range(T):
                jacobj[2 + i] += fsum(
                    (1 / n) * (w[i] - 1) * dkldivdy(kappa, alpha + np.dot(beta, w - 1) + np.dot(w, r)) for w, r in wrs)

            return jacobj

        def d2kldivdx2(x, y):
            return 1 / x

        def d2kldivdxdy(x, y):
            return -1 / y

        def d2kldivdy2(x, y):
            return x / y ** 2

        def hessnegdualobjective(x):
            kappa, alpha, beta = x[0], x[1], x[2:]

            hessobj = np.empty((2 + T, 2 + T))

            hessobj[0, 0] = fsum(
                (1 / n) * d2kldivdx2(kappa, alpha + np.dot(beta, w - 1) + np.dot(w, r)) for w, r in wrs)
            hessobj[0, 1] = fsum(
                (1 / n) * d2kldivdxdy(kappa, alpha + np.dot(beta, w - 1) + np.dot(w, r)) for w, r in wrs)
            for i in range(T):
                hessobj[0, 2 + i] = fsum(
                    (1 / n) * (w[i] - 1) * d2kldivdxdy(kappa, alpha + np.dot(beta, w - 1) + np.dot(w, r)) for w, r in
                    wrs)
            hessobj[1, 0] = hessobj[0, 1]
            hessobj[1, 1] = fsum(
                (1 / n) * d2kldivdy2(kappa, alpha + np.dot(beta, w - 1) + np.dot(w, r)) for w, r in wrs)
            for i in range(T):
                hessobj[1, 2 + i] = fsum(
                    (1 / n) * (w[i] - 1) * d2kldivdy2(kappa, alpha + np.dot(beta, w - 1) + np.dot(w, r)) for w, r in
                    wrs)
            hessobj[2:2 + T, 0] = hessobj[0, 2:2 + T]
            hessobj[2:2 + T, 1] = hessobj[1, 2:2 + T]
            for i in range(T):
                for j in range(T):
                    if i <= j:
                        hessobj[2 + i, 2 + j] = fsum((1 / n) * (w[i] - 1) * (w[j] - 1) * d2kldivdy2(kappa,
                                                                                                    alpha + np.dot(beta,
                                                                                                                   w - 1) + np.dot(
                                                                                                        w, r)) for w, r
                                                     in wrs)
                        hessobj[2 + j, 2 + i] = hessobj[2 + i, 2 + j]

            return hessobj

        x0 = np.hstack([1.0, 1.0, mlebetastar])

        def F(x=None, z=None):
            if x is None: return 0, matrix(x0)

            p = np.reshape(np.array(x), -1)
            kappa, alpha, beta = p[0], p[1], p[2:]

            if kappa < tiny or any(alpha + np.dot(beta, w - 1) + np.dot(w, r) < tiny for w, r in wrs):
                return None

            f = negdualobjective(p)
            jf = jacnegdualobjective(p)
            Df = matrix(jf).T
            if z is None: return f, Df
            hf = z[0] * hessnegdualobjective(p)
            H = matrix(hf, hf.shape)
            return f, Df, H

        G = np.vstack([np.array([1.0, 0.0] + [0.0] * T)] +
                      [np.hstack([[0.0], [wscale], wvec - wscale])
                       for wscale, wvec in constraintvecs()
                       ])
        h = np.array([tiny] + [0.0 for _ in constraintvecs()])

        soln = solvers.cp(F, G=-matrix(G, G.shape), h=-matrix(h), options={'show_progress': False})
        from pprint import pformat
        assert soln['status'] == 'optimal', pformat(soln)
        vhat = -soln['primal objective']
        xstar = np.reshape(np.array(soln['x']), -1)
        kappastar, alphastar, betastar = xstar[0], xstar[1], xstar[2:]

    # lame (!)
    #import torch
    #torchbeta = torch.from_numpy(betastar).float().detach()

    pstarfunc = lambda w, r: kappastar / (alphastar + betastar.dot(w - 1).item() + r.dot(w).item())
    #torchpstarfunc = lambda w, r: kappastar / (alphastar + torchbeta.dot(w - 1).item() + r.dot(w).item())
    return {'vhat': vhat, 'alpha': alphastar, 'beta': betastar, 'kappa': kappastar, 'pstarfunc': pstarfunc,
            'torchpstarfunc': None}

def print_gpus():
    ld = tf.config.list_logical_devices()
    for ldi in ld:
        print(ldi)


if __name__ == '__main__':
    import numpy as np
    N = 32
    np.random.seed(98052)
    w=np.random.poisson(1, size=(N,5))
    w = np.cumprod(w, axis=1)
    print(np.mean(w,axis=0))
    r = np.random.rand(N,5)
    wrs = list(zip(w,r))
    #print(wrs)
    print(np.sum(w*r)/N)
    print(pdislowerbound(wrs, 0, 20, 0.9))
    # print_gpus()
    # print(dopamine.name)
    #main()
