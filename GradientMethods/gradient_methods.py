# gradient_methods.py
"""Volume 2: Gradient Descent Methods.
<Sophie Gee>
<section 2>
<2/16/22>
"""
import scipy.optimize as opt
import scipy.linalg as la
import numpy as np
from autograd import grad
import autograd.numpy as anp
from matplotlib import pyplot as plt

# Problem 1
def steepest_descent(f, Df, x0, tol=1e-5, maxiter=100):
    """Compute the minimizer of f using the exact method of steepest descent.

    Parameters:
        f (function): The objective function. Accepts a NumPy array of shape
            (n,) and returns a float.
        Df (function): The first derivative of f. Accepts and returns a NumPy
            array of shape (n,).
        x0 ((n,) ndarray): The initial guess.
        tol (float): The stopping tolerance.
        maxiter (int): The maximum number of iterations to compute.

    Returns:
        ((n,) ndarray): The approximate minimum of f.
        (bool): Whether or not the algorithm converged.
        (int): The number of iterations computed.
    """
    #set the original variables
    xk = x0
    i = 1
    
    #start optimization
    while i < maxiter:
        ak = opt.minimize_scalar(lambda a: f(xk - a*Df(xk).T))
        xk_1 = xk - ak['x']*Df(xk).T

        #iterate until norm of derivative less than tolerance
        if la.norm(Df(xk), ord=np.inf) < tol:
            return xk, True, i
        
        #otherwise, keep iterating
        else:
            i += 1
            xk = xk_1

    return xk, False, i



# Problem 2
def conjugate_gradient(Q, b, x0, tol=1e-4):
    """Solve the linear system Qx = b with the conjugate gradient algorithm.

    Parameters:
        Q ((n,n) ndarray): A positive-definite square matrix.
        b ((n, ) ndarray): The right-hand side of the linear system.
        x0 ((n,) ndarray): An initial guess for the solution to Qx = b.
        tol (float): The convergence tolerance.

    Returns:
        ((n,) ndarray): The solution to the linear system Qx = b.
        (bool): Whether or not the algorithm converged.
        (int): The number of iterations computed.
    """
    #set intiial variables
    rk = (Q@x0) - b
    dk = -rk
    k = 0
    xk = x0

    #enter while loop from algorithm 5.1
    while la.norm(rk, ord=2) >= tol and k < len(Q):
        ak = (rk.T@rk)/(dk.T@Q@dk)
        xk_1 = xk + ak*dk
        rk_1 = rk + ak*Q@dk
        Bk_1 = (rk_1.T@rk_1)/(rk.T@rk)
        dk_1 = -rk_1 + (Bk_1*dk)
        k = k+1
        dk = np.copy(dk_1)
        xk = np.copy(xk_1)
        rk = np.copy(rk_1)
        #check to see why exited while loop, return accordingly
        if la.norm(rk, ord=2) < tol:
            return xk_1, True, k 

    return xk_1, False, k



# Problem 3
def nonlinear_conjugate_gradient(f, df, x0, tol=1e-5, maxiter=1000):
    """Compute the minimizer of f using the nonlinear conjugate gradient
    algorithm.

    Parameters:
        f (function): The objective function. Accepts a NumPy array of shape
            (n,) and returns a float.
        Df (function): The first derivative of f. Accepts and returns a NumPy
            array of shape (n,).
        x0 ((n,) ndarray): The initial guess.
        tol (float): The stopping tolerance.
        maxiter (int): The maximum number of iterations to compute.

    Returns:
        ((n,) ndarray): The approximate minimum of f.
        (bool): Whether or not the algorithm converged.
        (int): The number of iterations computed.
    """ 
    #set initial variables   
    r0 = -df(x0).T
    d0 = r0
    a0 = opt.minimize_scalar(lambda a: f(x0 + a * d0)).x
    xk = x0 + a0 * d0
    dk = d0
    rk = r0
    k = 1

    #enter while loop from 5.2 algortihm
    while True:
        #check for necessary convergent conditions
        if la.norm(rk, ord=2) <= tol:
            return xk, True, k
        elif k >= maxiter:
            return xk, False, k
        else:
            rp = rk
            dp = dk
            rk = -df(xk).T
            Bk = (rk.T @ rk) / (rp.T @ rp)
            dk = rk + Bk * dp
            ak = opt.minimize_scalar(lambda a: f(xk + a*dk)).x
            xn = xk + ak*dk
            xk = xn
            k = k + 1

# Problem 4
def prob4(filename="linregression.txt",
        x0=np.array([-3482258, 15, 0, -2, -1, 0, 1829])):
    """Use conjugate_gradient() to solve the linear regression problem with
    the data from the given file, the given initial guess, and the default
    tolerance. Return the solution to the corresponding Normal Equations.
    """
    #make the column vector of y, and vandermonde matrix of x
    x_y = np.loadtxt(filename)
    n = len(x_y)
    y = x_y[:, 0]
    x = np.ones((n, 1))
    x = np.column_stack((x, x_y[:, 1:]))

    #transpose A and multiply by b as well
    Q = x.T@x
    b = x.T@y

    #call function 2 to solve
    return(conjugate_gradient(Q, b, x0))[0]



# Problem 5
class LogisticRegression1D:
    """Binary logistic regression classifier for one-dimensional data."""

    def fit(self, x, y, guess):
        """Choose the optimal beta values by minimizing the negative log
        likelihood function, given data and outcome labels.

        Parameters:
            x ((n,) ndarray): An array of n predictor variables.
            y ((n,) ndarray): An array of n outcome variables.
            guess (array): Initial guess for beta.
        """
        #define the negative log likelihood function
        neg_log = lambda b: np.sum([np.log(1 + np.exp(-(b[0] + b[1]*x[i]))) + (1 - y[i])*(b[0] + b[1]*x[i]) for i in range(len(x))])

        #minimize it with function from problem 3
        self.b0, self.b1 = opt.fmin_cg(neg_log, guess, disp = False)


    def predict(self, x):
        """Calculate the probability of an unlabeled predictor variable
        having an outcome of 1.

        Parameters:
            x (float): a predictor variable with an unknown label.
        """
        #return probability
        return (1 / (1 + np.exp(-(self.b0 + (self.b1*x)))))


# Problem 6
def prob6(filename="challenger.npy", guess=np.array([20., -1.])):
    """Return the probability of O-ring damage at 31 degrees Farenheit.
    Additionally, plot the logistic curve through the challenger data
    on the interval [30, 100].

    Parameters:
        filename (str): The file to perform logistic regression on.
                        Defaults to "challenger.npy"
        guess (array): The initial guess for beta.
                        Defaults to [20., -1.]
    """
    #instantiate class from problem 5
    x, y = np.load(filename).T
    reg = LogisticRegression1D()

    #call fit
    reg.fit(x, y, guess)

    #create domain
    domain = np.linspace(30, 100, 200)
    y_1 = reg.predict(domain)

    plt.plot(domain, y_1)
    plt.scatter([31], [1], label= "P(damage) at Launch")
    plt.scatter(x, y, label = "Previous Damage")
    plt.legend()
    plt.title("Probability of O-ring Damage")

    plt.show()
    return reg.predict(31)
    


if __name__ == "__main__":
    print(prob6())



