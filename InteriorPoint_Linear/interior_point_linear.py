# interior_point_linear.py
"""Volume 2: Interior Point for Linear Programs.
<Sophie Gee>
<03/16/22                                                                       >
"""

import numpy as np
from pyrsistent import b
import scipy.linalg as la
from scipy.stats import linregress
from matplotlib import pyplot as plt
import numpy.ma as ma


# Auxiliary Functions ---------------------------------------------------------
def starting_point(A, b, c):
    """Calculate an initial guess to the solution of the linear program
    min c^T x, Ax = b, x>=0.
    Reference: Nocedal and Wright, p. 410.
    """
    # Calculate x, lam, mu of minimal norm satisfying both
    # the primal and dual constraints.
    B = la.inv(A @ A.T)
    x = A.T @ B @ b
    lam = B @ A @ c
    mu = c - (A.T @ lam)

    # Perturb x and s so they are nonnegative.
    dx = max((-3./2)*x.min(), 0)
    dmu = max((-3./2)*mu.min(), 0)
    x += dx*np.ones_like(x)
    mu += dmu*np.ones_like(mu)

    # Perturb x and mu so they are not too small and not too dissimilar.
    dx = .5*(x*mu).sum()/mu.sum()
    dmu = .5*(x*mu).sum()/x.sum()
    x += dx*np.ones_like(x)
    mu += dmu*np.ones_like(mu)

    return x, lam, mu

# Use this linear program generator to test your interior point method.


def randomLP(j, k):
    """Generate a linear program min c^T x s.t. Ax = b, x>=0.
    First generate m feasible constraints, then add
    slack variables to convert it into the above form.
    Parameters:
        j (int >= k): number of desired constraints.
        k (int): dimension of space in which to optimize.
    Returns:
        A ((j, j+k) ndarray): Constraint matrix.
        b ((j,) ndarray): Constraint vector.
        c ((j+k,), ndarray): Objective function with j trailing 0s.
        x ((k,) ndarray): The first 'k' terms of the solution to the LP.
    """
    A = np.random.random((j, k))*20 - 10
    A[A[:, -1] < 0] *= -1
    x = np.random.random(k)*10
    b = np.zeros(j)
    b[:k] = A[:k, :] @ x
    b[k:] = A[k:, :] @ x + np.random.random(j-k)*10
    c = np.zeros(j+k)
    c[:k] = A[:k, :].sum(axis=0)/k
    A = np.hstack((A, np.eye(j)))
    return A, b, -c, x


# Problems --------------------------------------------------------------------
def interiorPoint(A, b, c, niter=20, tol=1e-16, verbose=False):
    """Solve the linear program min c^T x, Ax = b, x>=0
    using an Interior Point method.

    Parameters:
        A ((m,n) ndarray): Equality constraint matrix with full row rank.
        b ((m, ) ndarray): Equality constraint vector.
        c ((n, ) ndarray): Linear objective function coefficients.
        niter (int > 0): The maximum number of iterations to execute.
        tol (float > 0): The convergence tolerance.

    Returns:
        x ((n, ) ndarray): The optimal point.
        val (float): The minimum value of the objective function.
    """
    m, n = A.shape

    x, lam, mu = starting_point(A, b, c)

    def vector_f(x, lam, mu):
        # create rows of F and row_stack to return F
        F_1 = A.T@lam + mu - c
        F_2 = A@x - b
        F_3 = np.diag(mu)@x

        F = np.hstack((F_1, F_2, F_3))
        return F

    def search_direction(F, x, mu):
        # change the third row of DF and solve using lu_solve
        DF = np.block([[np.zeros((n,n)), A.T, np.eye(n)],
                       [A, np.zeros((m,m)), np.zeros((m,n))],
                       [np.diag(mu), np.zeros((n,m)), np.diag(x)]])
        v = (x.T@mu) / n
        ove = np.hstack((np.zeros(n), np.zeros(m), (v/10.0)*np.ones(n)))
        b = -F + ove

        return la.lu_solve(la.lu_factor(DF), b), v

    def step_size(mu, x, lam, delta):
        # gather delta x, lam, mu
        delta_x = delta[:n]
        delta_lam = delta[n:-n]
        delta_mu = delta[-n:]

        # mask them and min
        if min(delta_mu) >= 0:
            alpha_max = 1
        else:
            new_mu = ma.masked_less(delta_mu, 0)
            alpha_max = np.min(-mu / new_mu)
        if min(delta_x) >= 0:
            delta_max= 1
        else:
            new_x = ma.masked_less(delta_x, 0)
            delta_max = np.min(-x / new_x)

        # final alpha and delta
        alpha = max(1, .95*alpha_max)
        d = max(1, .95*delta_max)

        x = x + d*delta_x
        lam = lam + alpha*delta_lam
        mu = mu + alpha*delta_mu

        return x, lam, mu

    # run the iteration niter times or until the duality measure is less than tol and return optimal values
    for _ in range(niter):

        F = vector_f(x, lam, mu)

        direction, measure = search_direction(F, x, mu)

        #checking measure with tolerance
        if measure <= tol:
            return x, c.T@x
        x, lam, mu = step_size(mu, x, lam, direction)
        

    return x, c.T@x


def leastAbsoluteDeviations(filename='simdata.txt'):
    """Generate and show the plot requested in the lab."""

    data = np.loadtxt(filename)

    #obtain m and n
    m = data.shape[0]
    n = data.shape[1] - 1

    c = np.zeros(3*m + 2*(n + 1))
    c[:m] = 1

    #x, y selected from data
    y = np.empty(2*m)
    y[::2] = -data[:, 0]
    y[1::2] = data[:, 0]

    x = data[:, 1:]

    #construct A
    A = np.ones((2*m, 3*m + 2*(n + 1)))
    A[::2, :m] = np.eye(m)
    A[1::2, :m] = np.eye(m)
    A[::2, m:m+n] = -x
    A[1::2, m:m+n] = x
    A[::2, m+n:m+2*n] = x
    A[1::2, m+n:m+2*n] = -x
    A[::2, m+2*n] = -1
    A[1::2, m+2*n+1] = -1
    A[:, m+2*n+2:] = -np.eye(2*m, 2*m)

    #calculate solution
    sol = interiorPoint(A, y, c, niter=10)[0]
    beta = sol[m:m+n] - sol[m+n:m+2*n]
    b = sol[m+2*n] - sol[m+2*n+1]
    slope, intercept = linregress(data[:,1], data[:,0])[:2]
    domain = np.linspace(0,10,200)

    #plot the least absolute deviations line
    plt.plot(domain, domain*slope + intercept, label = "OLS")
    plt.plot(domain, domain*beta + b, label = "LAD")
    plt.scatter(data[:, 1], data[:, 0], label= "Data")
    plt.title("Least Absolute Deviations Line")

    plt.show()

if __name__ == "__main__":
   leastAbsoluteDeviations()

    # from scipy.stats import linregress
    # slope, intercept = linregress(data[:,1], data[:,0])[:2]
    # domain = np.linspace(0,10,200)
    # plt.plot(domain, domain*slope + intercept)