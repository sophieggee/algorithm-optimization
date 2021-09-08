# python_intro.py
"""Python Essentials: Introduction to Python.
<Sophie Gee>
<MATH 321 Section 3>
<08/27/21>
"""
import math
import numpy as np
# Problem 1


def isolate(a, b, c, d, e):
    print(a, b, c, sep='     ', end=' ')
    print(d, e)

# Problem 2


def first_half(string):
    """return the first half of the passed in string"""
    x = len(string)
    y = x//2
    return string[:y]


def backward(string):
    """return string backward"""
    return string[::-1]

# Problem 3


def list_ops():
    """change the list 'bear, ant, cat, dog' to ''fox', 'hawk', 'dog', 'bearhunter''"""
    list = ["bear", "ant", "cat", "dog"]
    list.append("eagle")
    list[2] = "fox"
    list.pop(1)
    list = sorted(list, reverse=True)
    list[list.index("eagle")] = "hawk"
    list[-1] = list[-1]+"hunter"
    return list
print(list_ops)
# Problem 4


def alt_harmonic(n):
    """Return the partial sum of the first n terms of the alternating
    harmonic series. Use this function to approximate ln(2).
    """
    harmonic_sum = [(((-1)**(i+1))/i) for i in range(1, n+1)]
    return sum(harmonic_sum)


def prob5(A):
    """Make a copy of 'A' and set all negative entries of the copy to 0.
    Return the copy.

    Example:
        >>> A = np.array([-3,-1,3])
        >>> prob4(A)
        array([0, 0, 3])
    """
    new_A = A.copy()
    mask = new_A < 0
    new_A[mask] = 0
    return new_A


def prob6():
    """Define the matrices A, B, and C as arrays. Return the block matrix
                                | 0 A^T I |
                                | A  0  0 |,
                                | B  0  C |
    where I is the 3x3 identity matrix and each 0 is a matrix of all zeros
    of the appropriate size.
    """
    A = np.array([[0, 2, 4], [1, 3, 5]])
    B = np.array([[3, 0, 0], [3, 3, 0], [3, 3, 3]])
    C = np.array([[-2, 0, 0], [0, -2, 0], [0, 0, -2]])
    I = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    part_1 = np.vstack((np.zeros((3, 3)), A, B))
    part_2 = np.vstack((A.T, np.zeros((2, 2)), np.zeros((3, 2))))
    part_3 = np.vstack((I, np.zeros((2, 3)), C))
    return(np.concatenate((part_1, part_2, part_3), axis=1))
print(prob5())

def prob7(A):
    """Divide each row of 'A' by the row sum and return the resulting array.

    Example:
        >>> A = np.array([[1,1,0],[0,1,0],[1,1,1]])
        >>> prob6(A)
        array([[ 0.5       ,  0.5       ,  0.        ],
               [ 0.        ,  1.        ,  0.        ],
               [ 0.33333333,  0.33333333,  0.33333333]])
    """
    sums = A.sum(axis=1)
    A = A/sums.reshape(-1, 1)
    return(A)


def prob8():
    """Given the array stored in grid.npy, return the greatest product of four
    adjacent numbers in the same direction (up, down, left, right, or
    diagonally) in the grid.
    """
    grid = np.load("grid.npy")
    winner = 0
    for i in range(20):
        for j in range(17):
            winner = max(np.prod(grid[i, j:j+4]), winner)
            winner = max(np.prod(grid[j:j+4, i]), winner)
            if i < 17 and j < 17:
                diagonal_array_1 = [grid[i, j], grid[i+1,j+1], grid[i+2, j+2], grid[i+3, j+3]]
                winner = max(np.prod(diagonal_array_1), winner)
                diagonal_array_2 = [grid[19-i, j], grid[18-i, j+1], grid[17-i, j+2], grid[16-i, j+3]]
                winner = max(np.prod(diagonal_array_2), winner)
    return winner


if __name__ == '__main__':
    print(list_ops())
