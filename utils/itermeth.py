from scipy.linalg import lu
from numpy.linalg import pinv, norm
import numpy as np

def itermeth(A, b, x0, nmax, tol, P=None):
    """ITERMETH  General iterative method
     X = ITERMETH(A,B,X0,NMAX,TOL,P) attempts to solve the
     system of linear equations A*X=B for X. The N-by-N
     coefficient matrix A must be non-singular and the
     right hand side column vector B must have length
     N. If P='J' the Jacobi method is used, if P='G' the
     Gauss-Seidel method is selected. Otherwise, P is a
     N-by-N matrix that plays the role of a preconditioner
     for the dynamic Richardson method. Iterations
     stop when the ratio between the norm of the k-th
     residual and the norm of the initial residual is less
     than TOL, then ITER is the number of performed
     iterations. NMAX specifies the maximum
     number of iterations. If P is not defined, the
     dynamic unpreconditioned Richardson method
     is performed."""
    
    from scipy.linalg import lu
    from numpy.linalg import pinv, norm
    import numpy as np
    
    size = A.shape
    if size[0] == size[1]:
        n = size[0]
    
        if P == 'J':
            L = np.diag(np.diag(A))
            U = np.eye(n)
            beta, alpha = 1, 1
        elif P == 'G':
            L = np.tril(A)
            U = np.eye(n)
            beta, alpha = 1, 1
        elif P is None:
            L = np.eye(n)
            U = np.eye(n)
            beta = 0
        else:
            _, L, U = lu(A)
            beta = 0

        it = 0
        x = x0
        r = b - A @ x0
        r0 = norm(r)
        err = norm(r)
        while (err > tol) and (it < nmax):
            it += 1
            z = pinv(L)@r
            z = pinv(U)@z

            if beta == 0:
                alpha = (z.T @ r) / (z.T @ A @ z)

            x = x + alpha*z
            r = b - A@x
            err = norm(r)/r0

        return x, it
    
    else:
        print('Matrix A must be square')