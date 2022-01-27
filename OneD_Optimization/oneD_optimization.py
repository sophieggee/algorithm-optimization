# oneD_optimization.py
"""Volume 2: One-Dimensional Optimization.
<Sophie Gee>
<1/27/22>
"""
import numpy as np
from matplotlib import pyplot as plt
from scipy import optimize as opt

# Problem 1
def golden_section(f, a, b, tol=1e-5, maxiter=15):
    """Use the golden section search to minimize the unimodal function f.

    Parameters:
        f (function): A unimodal, scalar-valued function on [a,b].
        a (float): Left bound of the domain.
        b (float): Right bound of the domain.
        tol (float): The stopping tolerance.
        maxiter (int): The maximum number of iterations to compute.

    Returns:
        (float): The approximate minimizer of f.
        (bool): Whether or not the algorithm converged.
        (int): The number of iterations computed.
    """
    #set the initial minimizer approximation as the interval midpoint.
    x0 = (a + b)/2
    phi = (1 + np.sqrt(5))/2

    #Iterate only maxiter times at most.
    for i in range(maxiter):
        c = (b - a)/phi
        a_tilda = b - c
        b_tilda = a + c

        #Get new boundaries for the search interval.
        if f(a_tilda) <= f(b_tilda):
            b = b_tilda
        else:
            a = a_tilda
        
        #Set the minimizer approximation as the interval midpoint.
        x1 = (a + b)/2

        #Stop iterating if the approximation stops changing enough.
        if abs(x0 - x1) < tol:
            return x1, True, i
        
        x0 = x1

    return x1, False, maxiter
        

# Problem 2
def newton1d(df, d2f, x0, tol=1e-5, maxiter=15):
    """Use Newton's method to minimize a function f:R->R.

    Parameters:
        df (function): The first derivative of f.
        d2f (function): The second derivative of f.
        x0 (float): An initial guess for the minimizer of f.
        tol (float): The stopping tolerance.
        maxiter (int): The maximum number of iterations to compute.

    Returns:
        (float): The approximate minimizer of f.
        (bool): Whether or not the algorithm converged.
        (int): The number of iterations computed.
    """
    #recursively iterate
    for i in range(maxiter):
        x1 = x0 - (df(x0)/d2f(x0))

        #check the difference between x
        if abs(x0 - x1) < tol:
            return x1, True, i
        
        x0 = x1
    return x1, False, maxiter



# Problem 3
def secant1d(df, x0, x1, tol=1e-5, maxiter=15):
    """Use the secant method to minimize a function f:R->R.

    Parameters:
        df (function): The first derivative of f.
        x0 (float): An initial guess for the minimizer of f.
        x1 (float): Another guess for the minimizer of f.
        tol (float): The stopping tolerance.
        maxiter (int): The maximum number of iterations to compute.

    Returns:
        (float): The approximate minimizer of f.
        (bool): Whether or not the algorithm converged.
        (int): The number of iterations computed.
    """
    #iterate over maxiter
    for _ in range(maxiter):
        x_k_1 = (x0*df(x1) - x1*df(x0))/(df(x1) - df(x0))

        #check absolute difference
        if abs(x_k_1 - x1) < tol:
            return x1, True, _
        x0 = x1
        x1 = x_k_1

    return x1, False, maxiter


# Problem 4
def backtracking(f, Df, x, p, alpha=1, rho=.9, c=1e-4):
    """Implement the backtracking line search to find a step size that
    satisfies the Armijo condition.

    Parameters:
        f (function): A function f:R^n->R.
        Df (function): The first derivative (gradient) of f.
        x (float): The current approximation to the minimizer.
        p (float): The current search direction.
        alpha (float): A large initial step length.
        rho (float): Parameter in (0, 1).
        c (float): Parameter in (0, 1).

    Returns:
        alpha (float): Optimal step size.
    """
    #Compute these values only once.
    Dfp = Df(x).T@p
    fx = f(x)

    #check at each iteration
    while (f(x + alpha*p) > (fx + c*alpha*Dfp)):
        alpha = rho*alpha
    return alpha

if __name__ == "__main__":
    from scipy.optimize import linesearch
    from autograd import numpy as anp
    from autograd import grad
# Get a step size for f(x,y,z) = x^2 + y^2 + z^2.
    f = lambda x: x[0]**2 + x[1]**2 + x[2]**2
    Df = lambda x: np.array([2*x[0], 2*x[1], 2*x[2]])
    x = anp.array([150., .03, 40.])
    p = anp.array([-.5, -100., -4.5])
    phi = lambda alpha: f(x + alpha*p)
    dphi = grad(phi)
    alpha, _ = linesearch.scalar_search_armijo(phi, phi(0.), dphi(0.))
    print(alpha, backtracking(f, Df, x, p))