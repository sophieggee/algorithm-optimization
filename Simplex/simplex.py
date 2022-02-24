"""Volume 2: Simplex

<Sophie Gee>
<02/22/22>
<section 2>
"""

import numpy as np


# Problems 1-6
class SimplexSolver(object):
    """Class for solving the standard linear optimization problem

                        minimize        c^Tx
                        subject to      Ax <= b
                                         x >= 0
    via the Simplex algorithm.
    """
    # Problem 1
    def __init__(self, c, A, b):
        """Check for feasibility and initialize the dictionary.

        Parameters:
            c ((n,) ndarray): The coefficients of the objective function.
            A ((m,n) ndarray): The constraint coefficients matrix.
            b ((m,) ndarray): The constraint vector.

        Raises:
            ValueError: if the given system is infeasible at the origin.
        """
        if min(b) < 0:
            raise ValueError("Not feasible at the origin.")
        
        self._generatedictionary(c, A, b)



    # Problem 2
    def _generatedictionary(self, c, A, b):
        """Generate the initial dictionary.

        Parameters:
            c ((n,) ndarray): The coefficients of the objective function.
            A ((m,n) ndarray): The constraint coefficients matrix.
            b ((m,) ndarray): The constraint vector.
        """
        #create two 'columns' of D
        m = A.shape[0]
        Abar = np.column_stack((A, np.eye(m)))
        Cbar = np.concatenate((c, np.zeros(m)))

        self.dict = np.column_stack((
            np.concatenate(([0], b)),
            np.vstack((Cbar.T, -Abar))
        ))

    # Problem 3a
    def _pivot_col(self):
        """Return the column index of the next pivot column.
        """
        #choose first occurance of negative
        negative = [i for i, j in enumerate(self.dict[0]) if j < 0]
        if len(negative) == 1:
            return False
        return negative[1]
        

    # Problem 3b
    def _pivot_row(self, index):
        """Determine the row index of the next pivot row using the ratio test
        (Bland's Rule).
        """
        #access whole column
        column = self.dict[:,index]

        #terminate if the pivot column only has nonnegative entries
        if min(column) >= 0:
            return False

        #caluclate ratios
        ratios = []
        for i in range(1,len(self.dict)):
            #avoid dividing by 0
            #do not count nonnegative values, but still keep track of index
            if self.dict[i, index] >= 0:
                ratio = np.inf
            else:
                ratio = (- self.dict[i, 0]) / (self.dict[i, index])

            ratios.append(ratio) 

        #return row index
        return np.argmin(ratios) + 1

    # Problem 4
    def pivot(self):
        """Select the column and row to pivot on. Reduce the column to a
        negative elementary vector.
        """
        
        col = self._pivot_col()

        while col != False:

            row = self._pivot_row(col)

            #raise valueerror if problem is unbounded
            if row == False:
                raise ValueError("Problem is Unbounded")


            #first step:
            self.dict[row] = self.dict[row] / -(self.dict[row, col])
            

            #second step:
            for i, curr in enumerate(self.dict):
                if i == row:
                    continue
                #row operations
                else:
                    scale = - curr[col] / self.dict[row, col]
                    self.dict[i] += (scale * self.dict[row])

            #reset pivot column for checking
            col = self._pivot_col()
            

    # Problem 5
    def solve(self):
        """Solve the linear optimization problem.

        Returns:
            (float) The minimum value of the objective function.
            (dict): The basic variables and their values.
            (dict): The nonbasic variables and their values.
        """
        self.pivot()

        #look at first row, disregarding first element
        top_row = self.dict[0]
        dep_dict = {}
        ind_dict = {}

        for i, el in enumerate(top_row):
            if i == 0:
                minimal = el
            #get dependent variables and find their value  
            elif el == 0:
                neg_one = np.argmin(self.dict[:, i])
                dep_dict[i - 1] = self.dict[neg_one, 0]
            #get independent variables and set them to 0
            elif el > 0:
                ind_dict[i - 1] = 0
            else:
                print("BAD")

        return (minimal, dep_dict, ind_dict)


# Problem 6
def prob6(filename='productMix.npz'):
    """Solve the product mix problem for the data in 'productMix.npz'.

    Parameters:
        filename (str): the path to the data file.

    Returns:
        ((n,) ndarray): the number of units that should be produced for each product.
    """

    #create c, A, b
    data = np.load(filename)
    A = np.vstack((data["A"], np.identity(data["A"].shape[1])))
    c = -data["p"]
    b = np.append(data["m"], data["d"])

    #solve this system after creating c, A, b
    solver = SimplexSolver(c, A, b)
    sol = solver.solve()
    length = len(c)
    dep_dict = sol[1]

    #append only the first relevant, non-slack variable values
    vals = []
    for i in range(length):
        vals.append(dep_dict[i])
        
    return vals
