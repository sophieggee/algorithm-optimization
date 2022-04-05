# dynamic_programming.py
"""Volume 2: Dynamic Programming.
<Sophie Gee>
<Volume 2>
<3/31/22>
"""

import numpy as np
from matplotlib import pyplot as plt


def calc_stopping(N):
    """Calculate the optimal stopping time and expected value for the
    marriage problem.

    Parameters:
        N (int): The number of candidates.

    Returns:
        (float): The maximum expected value of choosing the best candidate.
        (int): The index of the maximum expected value.
    """
    #create the dictionary of V with N values
    V = {N : 0}

    for t in range(N - 1, 0, -1):

        #increment accordingly
        V[t] = max(V[t + 1], t/(t + 1)*V[t + 1] + 1/N)

    return V[max(V, key=V.get)], max(V, key=V.get)
    



# Problem 2
def graph_stopping_times(M):
    """Graph the optimal stopping percentage of candidates to interview and
    the maximum probability against M.

    Parameters:
        M (int): The maximum number of candidates.

    Returns:
        (float): The optimal stopping percent of candidates for M.
    """
    #create percentages and maximum probability lists to graph
    percentages = []
    max_prob = []
    domain = np.arange(3, M)

    #calculate the maximum probability dictionaries at each step
    for r in range(3, M):
        prob, stopping  = calc_stopping(r)
        perc = stopping / r
        percentages.append(perc)
        max_prob.append(prob)

    #plot both lines
    plt.plot(domain, percentages, label = "percentages")
    plt.plot(domain, max_prob, label = "maximum probability")
    plt.legend()
    plt.show()

    return percentages[-1]



# Problem 3
def get_consumption(N, u=lambda x: np.sqrt(x)):
    """Create the consumption matrix for the given parameters.

    Parameters:
        N (int): Number of pieces given, where each piece of cake is the
            same size.
        u (function): Utility function.

    Returns:
        C ((N+1,N+1) ndarray): The consumption matrix.
    """
    #create consumption matrix, and w
    C = np.zeros((N + 1, N + 1))
    w = [i / N for i in range(N + 1)]

    #build the consumption matrix through rows and columns
    for i in range(N + 1):
        for j in range(i):
            C[i][j] = u((w[i] - w[j]))

    return C



# Problems 4-6
def eat_cake(T, N, B, u=lambda x: np.sqrt(x)):
    """Create the value and policy matrices for the given parameters.

    Parameters:
        T (int): Time at which to end (T+1 intervals).
        N (int): Number of pieces given, where each piece of cake is the
            same size.
        B (float): Discount factor, where 0 < B < 1.
        u (function): Utility function.

    Returns:
        A ((N+1,T+1) ndarray): The matrix where the (ij)th entry is the
            value of having w_i cake at time j.
        P ((N+1,T+1) ndarray): The matrix where the (ij)th entry is the
            number of pieces to consume given i pieces at time j.
    """
    #instantiate A and P as a matrix of zeros, and w as before 
    A = np.zeros((N + 1, T + 1))
    P = np.zeros((N + 1, T + 1))
    w = [i / N for i in range(N + 1)]

    #fill out last columns of P and A
    for i in range(N + 1):
        A[i, T] = u(w[i])
        P[i, T] = w[i]
    
    #iterate through t's backwards and iteratively build CV, choose max column of CV
    #to set A[:, t] to the maximum column of CV and set P[i, t] to w[i] - w[j]
    for t in range(T -1, -1, -1):
        CV = np.zeros((N + 1, N + 1))
        for i in range(N + 1):
            for j in range(i + 1):
                CV[i, j] = (u(w[i] - w[j]) + B*A[j, t + 1])
            P[i, t] = w[i] - w[np.argmax(CV[i, :])]
        A[:, t] = np.max(CV, axis = 1)
    
    return A, P



# Problem 7
def find_policy(T, N, B, u=np.sqrt):
    """Find the most optimal route to take assuming that we start with all of
    the pieces. Show a graph of the optimal policy using graph_policy().

    Parameters:
        T (int): Time at which to end (T+1 intervals).
        N (int): Number of pieces given, where each piece of cake is the
            same size.
        B (float): Discount factor, where 0 < B < 1.
        u (function): Utility function.

    Returns:
        ((T,) ndarray): The matrix describing the optimal percentage to
            consume at each time.
    """
    P = eat_cake(T = T, N = N, B = B, u = u)[1]

    #optimal amount of cake to eat at 0: p[N, 0] / slice size which is 1/N so multiply by N
    opt = np.zeros(T + 1)
    opt[0] = P[N, 0]

    for t in range(T + 1):
        opt_sum = sum([opt[i] for i in range(t)])
        ind = N - (N * opt_sum)
        opt[t] = P[int(ind), t]
    
    return opt


if __name__ == "__main__":
    print(graph_stopping_times(1000))