import numpy as np
import cvxopt
class LogisticRegression:
    """
        Logistic regression from scratch using the weighted Ridge regression method
        
        """
    
    def __init__(self, lambd=0.1):
        self.lambd = lambd
        
    def solveWRR(self, y, X, w, lam):
        n, p = X.shape
        assert (len(y) == len(w) == n)
        beta = np.linalg.solve((np.diag(np.sqrt(w)).dot(X)).T.dot(np.diag(np.sqrt(w)).dot(X)) + n*lam*np.eye(p),\
                               (np.diag(np.sqrt(w)).dot(X)).T.dot(np.diag(np.sqrt(w)).dot(y)))
        return (beta)
    
    def fit(self, y, X, max_iter=100):
        lam = self.lambd
        sigma = lambda a: 1/(1 + np.exp(-a))
        X = np.hstack([np.ones(len(X)).reshape(-1,1), X])
        n, p = X.shape
        assert (len(y) == n)
        beta_old = np.zeros(p)
        # Hint: Use IRLS
        max_iter = max_iter
        eps = 1e-12
        for i in range(max_iter):
            f = X.dot(beta_old)
            w = sigma(f)*sigma(-f)
            z = f + y/sigma(f*y)
            beta = self.solveWRR(z, X, w, 2*lam)
            if np.linalg.norm(beta-beta_old) < eps:
                break
            beta_old = beta
        self.beta = beta
        
    def predict(self, Xtest):
        Xtest = np.hstack([np.ones(len(Xtest)).reshape(-1,1), Xtest])
        probas_pred = Xtest.dot(self.beta)
        pred = np.sign(probas_pred)
        return pred
    
    def compute_accuracy(self, y, pred):
        correct = 0
        for idx in range(len(pred)):
            if pred[idx] == y[idx]:
                correct += 1.
        return correct/len(y)

## SVM
solver = 'cvxopt'
class MySVM:
    def __init__(self, use_kernel=True, degree=2, std=10., kernels=['poly','gauss'], combine_kernels = 'prod', C=10.):
        self.solve_qp = {'cvxopt': self.cvxopt_qp}[solver]
        self.use_kernel = use_kernel
        self.degree = degree
        self.std = std
        self.kernels = kernels
        self.combine_kernels = combine_kernels
        self.alpha = None
        self.C = C
        self.mismatch = 2
        self.w, self.b = None, None
    
    def cvxopt_qp(self, P, q, G, h, A, b):
        P = .5 * (P + P.T)
        cvx_matrices = [
            cvxopt.matrix(M) if M is not None else None for M in [P, q, G, h, A, b] 
        ]
        solution = cvxopt.solvers.qp(*cvx_matrices)
        return np.array(solution['x']).flatten()
    
    def get_primal_from_dual(self, alpha, Xtr, y, hard_margin=False, C=None, tol=1e-5, verbose = False):
        # w parameter in vectorized form
        if not self.use_kernel:
            w = (alpha*y).T.dot(Xtr).flatten()
        sv = (alpha > tol)
        # If soft margin, also filter out points with alpha == C
        if not hard_margin:
            if C is None:
                raise ValueError('C must be defined in soft margin mode')
            sv = np.logical_and(sv, (C - alpha > tol))
        b = y[sv] - Xtr[sv].dot(w)
        b = b[0]

        #Display results
        if verbose:
            print('Alphas = {}'.format(alpha[sv]))
            print('Number of support vectors = {}'.format(sv.sum()))
            print('w = {}'.format(w))
            print('b = {}'.format(b))

        return w, b
    
    def svm_dual_soft_to_qp(self, Xtr, y, C=1):
        n, p = Xtr.shape
        assert (len(y) == n)
        # Dual formulation, soft margin
        if self.use_kernel:
#             K = self.kernel(Xtr, Xtr)
#             Ky = np.diag(y).dot(K)
#             P = 0.5 * Ky.dot(Ky.T)
            self.K = self.kernel(Xtr, Xtr)
            P = np.diag(y).dot(self.K).dot(np.diag(y))
        else:
            Xy = np.diag(y).dot(Xtr)
            P = Xy.dot(Xy.T)
        # As a regularization, we add epsilon * identity to P
        eps = 1e-12
        P += eps * np.eye(n)
        q = - np.ones(n)
        G = np.vstack([-np.eye(n), np.eye(n)])
        h = np.hstack([np.zeros(n), C * np.ones(n)])
        A = y[np.newaxis, :]
        b = np.array([0.])
        return P, q, G, h, A, b
    
    def kernel(self, X1, X2):
        d, sig = self.degree, self.std
        if self.use_kernel and len(self.kernels) == 2:
            if self.combine_kernels == 'prod':
                X2_norm = np.sum(X2 ** 2, axis = -1)
                X1_norm = np.sum(X1 ** 2, axis = -1)
                gamma = 1/(2*sig ** 2)
                K1 = np.sqrt(np.exp(- gamma * (X1_norm[:, None] + X2_norm[None, :] - 2 * np.dot(X1, X2.T))))
                K2 = (1.+X1.dot(X2.T))**d
                K = K1*K2
            elif self.combine_kernels == 'sum':
                X2_norm = np.sum(X2 ** 2, axis = -1)
                X1_norm = np.sum(X1 ** 2, axis = -1)
                gamma = 1./(2 * sig ** 2)
                K1 = np.sqrt(np.exp(- gamma * (X1_norm[:, None] + X2_norm[None, :] - 2 * np.dot(X1, X2.T))))
                K2 = (1.+X1.dot(X2.T))**d
                K = K1+K2
        else:
            if self.kernels[0] == 'poly':
                K = (1.+X1.dot(X2.T))**d
            else:
                X2_norm = np.sum(X2 ** 2, axis = -1)
                X1_norm = np.sum(X1 ** 2, axis = -1)
                gamma = 1 /(2*sig ** 2)
                K = np.sqrt(np.exp(- gamma * (X1_norm[:, None] + X2_norm[None, :] - 2 * np.dot(X1, X2.T))))
                #K = np.exp(- gamma * (X1_norm[:, None] + X2_norm[None, :] - 2 * np.dot(X1, X2.T)))
        return K

    def fit_no_kernel(self, X_tr, y_tr):
        self.use_kernel=False
        self.alpha = self.solve_qp(*self.svm_dual_soft_to_qp(X_tr, y_tr, C=self.C))
        self.w, self.b = self.get_primal_from_dual(self.alpha, X_tr, y_tr, hard_margin=False, C=self.C, tol=1e-5)
        
    def predict_no_kernel(self, X):
        pred = np.sign(X.dot(self.w)+self.b)
        return np.array(pred)
    
    def fit_kernel(self, X, y, tol=1e-8):
        n, p = X.shape
        assert (n == len(y))
        self.X_train = X
        self.y_train = y
        # Kernel matrix
        K = self.kernel(X, X)
        self.K = K
        # Solve dual problem
        self.alpha = self.solve_qp(*self.svm_dual_soft_to_qp(self.X_train, self.y_train, C=self.C))        
        # Compute support vectors and bias b
        sv = (self.alpha > tol)
        sv = np.logical_and(sv, (self.C - self.alpha > tol))
        self.bias = (y[sv] - K[sv].dot(self.alpha*y)).mean()
        return self
    
    def Kernel_predict(self, Xt):
        prediction = self.kernel(Xt, self.X_train).dot(self.alpha*self.y_train) + self.bias
        prediction = np.sign(prediction)
        prediction = np.array(prediction, dtype=int)
        return prediction
    
    def compute_accuracy(self, y_pred, y):
        correct = 0
        for idx in range(len(y_pred)):
            if y_pred[idx] == y[idx]:
                correct += 1.
        return correct/len(y)