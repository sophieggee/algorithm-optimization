# cvxpy_intro.py
"""Volume 2: Intro to CVXPY.
<Sophie Gee>
<3/9/22>
"""
import cvxpy as cp
import numpy as np

def prob1():
    """Solve the following convex optimization problem:

    minimize        2x + y + 3z
    subject to      x  + 2y         <= 3
                         y   - 4z   <= 1
                    2x + 10y + 3z   >= 12
                    x               >= 0
                          y         >= 0
                                z   >= 0

    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (float)
    """
    #initialize the objective and declare x with its size and sign, initialize c
    x = cp.Variable(3, nonneg = True)
    c = np.array([2, 1, 3])
    objective = cp.Minimize(c.T @ x)

    #write the constraints
    G = np.array([[1, 2, 0],[0, 1, -4]])
    P = np.array([[2, 10, 3], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    q = np.array([12, 0, 0, 0])
    h = np.array([3, 1])

    constraints = [G @ x <= h, P @ x >= q]

    #assemble the problem and solve it:
    problem = cp.Problem(objective, constraints)
    value = problem.solve()

    #return optimizer and optimal value
    return x.value, value


# Problem 2
def l1Min(A, b):
    """Calculate the solution to the optimization problem

        minimize    ||x||_1
        subject to  Ax = b

    Parameters:
        A ((m,n) ndarray)
        b ((m, ) ndarray)

    Returns:
        The optimizer x (ndarray)
        The optimal value (float)
    """
    #get n for initializing x and finding norm
    n = np.shape(A)[1]
    x = cp.Variable(n, nonneg = True)
    objective = cp.Minimize(cp.norm(x, 1))

    #initialize constraint Ax = b
    constraint = [A @ x == b]

    #assemble the problem and solve it:
    problem = cp.Problem(objective, constraint)
    
    #return optimizer and optimal value
    value = problem.solve()

    return x.value, value

# Problem 3
def prob3():
    """Solve the transportation problem by converting the last equality constraint
    into inequality constraints.

    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (float)
    """
    #initializing x and objective
    x = cp.Variable(6, nonneg = True)
    c = np.array([4, 7, 6, 8, 8, 9])
    objective = cp.Minimize(c.T @ x)

    #initialize constraint Ax == b
    A = np.array([[1, 1, 0, 0, 0, 0],
                  [0, 0, 1, 1, 0, 0],
                  [0, 0, 0, 0, 1, 1], 
                  [0, -1, 0, -1, 0, -1],
                  [-1, 0, -1, 0, -1, 0]])
    b = np.array([7, 2, 4, -8, -5])
    constraint = [x >= 0, A @ x == b]

    #assemble the problem and solve it:
    problem = cp.Problem(objective, constraint)
    
    #return optimizer and optimal value
    value = problem.solve()
    
    return x.value, value


# Problem 4
def prob4():
    """Find the minimizer and minimum of

    g(x,y,z) = (3/2)x^2 + 2xy + xz + 2y^2 + 2yz + (3/2)z^2 + 3x + z

    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (float)
    """
    #define Q, and r
    Q = np.array([[3, 2, 1], [2, 4, 2], [1, 2, 3]])
    r = np.array([3, 0, 1])
    x = cp.Variable(3)

    #use cp to solve quadratic
    prob = cp.Problem(cp.Minimize(.5 * cp.quad_form(x, Q) + r.T @ x))

    value = prob.solve()
    return x.value, value

#need help with problem 5!!
# Problem 5
def prob5(A, b):
    """Calculate the solution to the optimization problem
        minimize    ||Ax - b||_2
        subject to  ||x||_1 == 1
                    x >= 0
    Parameters:
        A ((m,n), ndarray)
        b ((m,), ndarray)
        
    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (float)
    """
    #get n for initializing x and finding norm
    n = np.shape(A)[1]
    x = cp.Variable(n, nonneg = True)
    objective = cp.Minimize(cp.norm(A @ x - b, 2))

    #initialize constraint 1-norm and nonneg values
    constraint = [ x.T@np.ones(n) == 1]

    #assemble the problem and solve it:
    problem = cp.Problem(objective, constraint)
    
    #return optimizer and optimal value
    value = problem.solve()

    return x.value, value


# Problem 6
def prob6():
    """Solve the college student food problem. Read the data in the file 
    food.npy to create a convex optimization problem. The first column is 
    the price, second is the number of servings, and the rest contain
    nutritional information. Use cvxpy to find the minimizer and primal 
    objective.
    
    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (float)
    """	 
    food = np.load("food.npy", allow_pickle = True)
    p = food[:, 0]
    serve = food[:, 1]
    c = np.array([serve[i] * food[:, 2][i] for i in range(18)])
    f = np.array([serve[i] * food[:, 3][i] for i in range(18)])
    s_hat = np.array([serve[i] * food[:, 4][i] for i in range(18)])
    c_hat = np.array([serve[i] * food[:, 5][i] for i in range(18)])
    f_hat = np.array([serve[i] * food[:, 6][i] for i in range(18)])
    p_hat = np.array([serve[i] * food[:, 7][i] for i in range(18)])

    #get n for initializing x and finding norm
    x = cp.Variable(18, nonneg = True)
    objective = cp.Minimize(p.T@x)

    #initialize constraint Ax = b
    constraint = [c.T@x <= 2000,
                    f.T@x <= 65,
                    s_hat.T@x <= 50,
                    c_hat.T@x >= 1000,
                    f_hat.T@x >= 25,
                    p_hat.T@x >= 46,
                    x >= 0]

    #assemble the problem and solve it:
    problem = cp.Problem(objective, constraint)
    
    #return optimizer and optimal value
    value = problem.solve()

    return x.value, value

if __name__ == "__main__":
    A = np.array([[1, 2, 1, 1], [0, 3, -2, -1]])
    b = np.array([7, 4])

    #print(prob5(A, b))
    print(prob6())