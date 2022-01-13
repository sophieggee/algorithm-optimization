# polynomial_interpolation.py
"""Volume 2: Polynomial Interpolation.
<Sophie Gee>
<Vol2>
<1/13/22>
"""
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import BarycentricInterpolator
import scipy.linalg as la
from numpy.fft import fft 

# Problems 1 and 2
def lagrange(xint, yint, points):
    """Find an interpolating polynomial of lowest degree through the points
    (xint, yint) using the Lagrange method and evaluate that polynomial at
    the specified points.

    Parameters:
        xint ((n,) ndarray): x values to be interpolated.
        yint ((n,) ndarray): y values to be interpolated.
        points((m,) ndarray): x values at which to evaluate the polynomial.

    Returns:
        ((m,) ndarray): The value of the polynomial at the specified points.
    """
    n = len(xint)
    m = len(points)

    denom = np.zeros(n)
    numer = np.zeros(n).astype(object)
    matrix = np.zeros((n,m))

    for j in range(n):
        #compute denominator:
        denom[j] = np.product([xint[j]-xint[k] for k in range(n) if k!= j])
        numer[j] = lambda x: np.product([x-xint[k] for k in range(n) if k!= j])

        #Lj = num/den
        for k in range(m):
            matrix[j][k] = (numer[j](points[k]))/denom[j]

    #matrix multiplication
    return matrix.T@yint



# Problems 3 and 4
class Barycentric:
    """Class for performing Barycentric Lagrange interpolation.

    Attributes:
        w ((n,) ndarray): Array of Barycentric weights.
        n (int): Number of interpolation points.
        x ((n,) ndarray): x values of interpolating points.
        y ((n,) ndarray): y values of interpolating points.
    """

    def __init__(self, xint, yint):
        """Calculate the Barycentric weights using initial interpolating points.

        Parameters:
            xint ((n,) ndarray): x values of interpolating points.
            yint ((n,) ndarray): y values of interpolating points.
        """
        #initialize important variables
        self.n = len(yint)
        self.x = xint
        self.y = yint

        #compute weights
        w = np.ones(n)                  
        # Array for storing barycentric weights.
        # Calculate the capacity of the interval.
        C = (np.max(xint) - np.min(xint)) / 4
        shuffle = np.random.permutation(n-1)
        for j in range(n):
            temp = (xint[j] - np.delete(xint, j)) / C
            temp = temp[shuffle]        # Randomize order   of product.
            w[j] /= np.product(temp)

        self.w = w

    def __call__(self, points):
        """Using the calcuated Barycentric weights, evaluate the interpolating polynomial
        at points.

        Parameters:
            points ((m,) ndarray): Array of points at which to evaluate the polynomial.

        Returns:
            ((m,) ndarray): Array of values where the polynomial has been computed.
        """
        def p(x):
            n = np.array([self.w[j]*self.y[j]/(x-self.x[j] + 10e-14) for j in range(self.n)])
            d = n = np.array([self.w[j]/(x-self.x[j] + 10e-14) for j in range(self.n)])
            return np.sum(n)/np.sum(d)

        #matrix multiplication
        return [p(x) for x in points]

    # Problem 4
    def add_weights(self, xint, yint):
        """Update the existing Barycentric weights using newly given interpolating points
        and create new weights equal to the number of new points.

        Parameters:
            xint ((m,) ndarray): x values of new interpolating points.
            yint ((m,) ndarray): y values of new interpolating points.
        """

        #update X and Y
        self.x = np.append(self.x, xint)
        self.y = np.append(self.y, yint)
        self.n = len(self.x)

        #compute weights
        w = np.ones(self.n)                  
        # Array for storing barycentric weights.
        # Calculate the capacity of the interval.
        C = (np.max(xint) - np.min(xint)) / 4
        shuffle = np.random.permutation(self.n-1)
        for j in range(n):
            temp = (xint[j] - np.delete(xint, j)) / C
            temp = temp[shuffle]        # Randomize order   of product.
            w[j] /= np.product(temp)
        self.w = w

# Problem 5
def prob5():
    """For n = 2^2, 2^3, ..., 2^8, calculate the error of intepolating Runge's
    function on [-1,1] with n points using SciPy's BarycentricInterpolator
    class, once with equally spaced points and once with the Chebyshev
    extremal points. Plot the absolute error of the interpolation with each
    method on a log-log plot.
    """
    #define domain x of 400 equally spaced points [-1,1]
    domain = np.linspace(-1,1,400)


    powers = [2**i for i in range(2,9)]
    err_z = np.zeros(7)
    err_e = np.zeros(7)
    i = 0
    for n in powers:
        runge = lambda x: 1 / ( 1 + 25 * x ** 2)
        x = np.linspace(-1, 1, n)
        poly = BarycentricInterpolator(x, runge(x))
        err_z[i] = la.norm(runge(domain) -poly(domain), ord=np.inf)

        X = np.array([np.cos(((2*k-1)/(2*n))*np.pi) for k in range(1,n+1)])

        poly_2 = BarycentricInterpolator(X, runge(X))
        err_e[i] = la.norm(runge(domain) -poly_2(domain), ord=np.inf)
        i+=1

    plt.loglog(powers, err_z, label = "errors of equally spaced points")
    plt.loglog(powers, err_e, label="errors of extremals")
    plt.legend()
    plt.show()




# Problem 6
def chebyshev_coeffs(f, n):
    """Obtain the Chebyshev coefficients of a polynomial that interpolates
    the function f at n points.

    Parameters:
        f (function): Function to be interpolated.
        n (int): Number of points at which to interpolate.

    Returns:
        coeffs ((n+1,) ndarray): Chebyshev coefficients for the interpolating polynomial.
    """

    #calculate checbyshev coefficients at given n
    y = np.cos((np.pi * np.arange(2*n))/n)
    samples = f(y)

    coeff = np.real(fft(samples))[:n+1] / n
    coeff[0] = coeff[0]/2
    coeff[n] = coeff[n]/2

    return coeff


# Problem 7
def prob7(n):
    """Interpolate the air quality data found in airdata.npy using
    Barycentric Lagrange interpolation. Plot the original data and the
    interpolating polynomial.

    Parameters:
        n (int): Number of interpolating points to use.
    """
    data = np.load("airdata.npy")
    print(np.shape(data))

    fx = lambda a, b, n: .5*(a+b + (b-a) * np.cos(np.arange(n+1) * np.pi / n))
    a, b = 0, 366 - 1/24
    domain = np.linspace(0, b, 8784)
    points = fx(a, b, n)
    temp = np.abs(points - domain.reshape(8784, 1))
    temp2 = np.argmin(temp, axis=0)
    
    poly = BarycentricInterpolator(domain[temp2], data[temp2])
    a1 = plt.subplot(121)
    a1.plot(domain, poly(domain) ,label="baryncentric interpolation")
    plt.legend()
    a2 = plt.subplot(122)
    a2.plot(domain, data, label="original function")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    """n = 5
    runge = lambda x: 1 / ( 1 + 25 * x ** 2)
    x = np.linspace(-1, 1, n)
    y = runge(x)
    domain = np.linspace(-1, 1, 100)
    output = Barycentric(x, y)
    plt.plot(domain, runge(domain), 'c-', label='Original')
    plt.plot(domain, output(domain), 'r-', label='Interpolation')
    plt.legend(loc='best')
    plt.show() """

    prob7(50)
