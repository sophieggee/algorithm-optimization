# quassian_quadrature.py
"""Volume 2: Gaussian Quadrature.
<Sophie Gee>
<amber>
<1/20/22>
"""
from scipy.stats import norm
import numpy as np
from scipy import linalg as la
from scipy.integrate import quad, nquad
from matplotlib import pyplot as plt

class GaussianQuadrature:
    """Class for integrating functions on arbitrary intervals using Gaussian
    quadrature with the Legendre polynomials or the Chebyshev polynomials.
    """
    # Problems 1 and 3
    def __init__(self, n, polytype="legendre"):
        """Calculate and store the n points and weights corresponding to the
        specified class of orthogonal polynomial (Problem 3). Also store the
        inverse weight function w(x)^{-1} = 1 / w(x).

        Parameters:
            n (int): Number of points and weights to use in the quadrature.
            polytype (string): The class of orthogonal polynomials to use in
                the quadrature. Must be either 'legendre' or 'chebyshev'.

        Raises:
            ValueError: if polytype is not 'legendre' or 'chebyshev'.
        """
        #check label and store as an attribute if not "legendre" or "chebyshev"
        self.n = n
        polytypes = ["legendre", "chebyshev"]
        if polytype not in polytypes:
            raise ValueError("Must be legendre or chebyshev")
        else:
            self.label = polytype
        
        #store weight functions as attribute
        if polytype == "legendre":
            self.w_x = lambda x: 1
        else:
            self.w_x = lambda x: np.sqrt(1-x**2)

        #store weights and points
        self.p, self.w = self.points_weights(n)
        

    # Problem 2
    def points_weights(self, n):
        """Calculate the n points and weights for Gaussian quadrature.

        Parameters:
            n (int): The number of desired points and weights.

        Returns:
            points ((n,) ndarray): The sampling points for the quadrature.
            weights ((n,) ndarray): The weights corresponding to the points.
        """
        #check cases and proceed accordingly
        if self.label == "legendre":
            B = [(k**2)/((4*k**2)-1) for k in range(1,n)]
        elif self.label == "chebyshev":
            B = np.full(n-1, 1/4)
            B[0] = 1/2
        
        #construct Jacobi matrix
        J = np.diag(np.sqrt(B), k=1)
        J1 = np.diag(np.sqrt(B), k=-1)
        Jacobi = J+J1

        #compute eigenvalues and vectors of J
        X = la.eig(Jacobi)[0]
        V = la.eig(Jacobi)[1]
    
        #compute the weights
        if self.label == "legendre":
            w = 2*V[0,:]**2
        elif self.label == "chebyshev":
            w = np.pi*V[0,:]**2

        return X, w

    # Problem 3
    def basic(self, f):
        """Approximate the integral of a f on the interval [-1,1]."""
        #create g without division
        g = lambda x: self.w_x(x)*f(x)
        
        #construct g with correct points from preceeding problem
        G = g(self.p)

        return np.dot(self.w, G)

    # Problem 4
    def integrate(self, f, a, b):
        """Approximate the integral of a function on the interval [a,b].

        Parameters:
            f (function): Callable function to integrate.
            a (float): Lower bound of integration.
            b (float): Upper bound of integration.

        Returns:
            (float): Approximate value of the integral.
        """
        #define h to be changing bounds and use previous methods
        h = lambda x: f(((b-a)/2)*x+(a+b)/2)

        #use method from problem 3
        return ((b-a)/2)*self.basic(h)


    # Problem 6.
    def integrate2d(self, f, a1, b1, a2, b2):
        """Approximate the integral of the two-dimensional function f on
        the interval [a1,b1]x[a2,b2].

        Parameters:
            f (function): A function to integrate that takes two parameters.
            a1 (float): Lower bound of integration in the x-dimension.
            b1 (float): Upper bound of integration in the x-dimension.
            a2 (float): Lower bound of integration in the y-dimension.
            b2 (float): Upper bound of integration in the y-dimension.

        Returns:
            (float): Approximate value of the integral.
        """
        #take inside integral
        h = lambda x, y: f(((b1-a1)/2)*x+((a1+b1)/2), ((b2-a2)/2)*y+((a2+b2)/2))

        #define g 
        g = lambda x, y: self.w_x(x)*self.w_x(y)*h(x,y)
        est = 0

        #double summations
        for i in range(self.n):
            for j in range(self.n):
                est = est + self.w[i]*self.w[j]*g(self.p[i], self.p[j])
        return (((b1-a1)*(b2-a2))/4)*est


# Problem 5
def prob5():
    """Use scipy.stats to calculate the "exact" value F of the integral of
    f(x) = (1/sqrt(2 pi))e^((-x^2)/2) from -3 to 2. Then repeat the following
    experiment for n = 5, 10, 15, ..., 50.
        1. Use the GaussianQuadrature class with the Legendre polynomials to
           approximate F using n points and weights. Calculate and record the
           error of the approximation.
        2. Use the GaussianQuadrature class with the Chebyshev polynomials to
           approximate F using n points and weights. Calculate and record the
           error of the approximation.
    Plot the errors against the number of points and weights n, using a log
    scale for the y-axis. Finally, plot a horizontal line showing the error of
    scipy.integrate.quad() (which doesn’t depend on n).
    """
    #define f 
    f = lambda x: (1/np.sqrt(2*np.pi))*np.exp(-(x**2)/2)

    #calculate real value of integral
    true_val = norm.cdf(2)-norm.cdf(-3)

    #repeat the following experiment for various n values
    leg_err = []
    cheb_err = []
    n = np.linspace(5,50,10)

    for i in n:
        #calculate the values and errors with legendre
        poly = GaussianQuadrature(int(i), "legendre")
        est_val = poly.integrate(f,-3,2)
        leg_err.append(np.abs(est_val-true_val))

        #calculate the values and errors with chebyshev
        poly = GaussianQuadrature(int(i), "chebyshev")
        est_val = poly.integrate(f,-3,2)
        cheb_err.append(np.abs(est_val-true_val))

    #horizontal line of error for scipy.integrate
    
    error = np.abs(quad(f, -3,2)[0]-true_val)
    err = np.full(10,error)

    #plot with y axis as log
    plt.semilogy(n, leg_err, label= "Legendre Error")
    plt.semilogy(n, cheb_err, label= "Chebyshev Error")
    plt.semilogy(n, err, label= "Scipy Quad Error")
    plt.legend()
    plt.title("Errors v N Value")
    plt.show()

if __name__ == "__main__":
    prob5()

