# policy_iteration.py
"""Volume 2: Policy Function Iteration.
<Sophie Gee>
<Volume 2>
<04/07/22>
"""

from re import I
import numpy as np
import scipy.linalg as la
import gym
from gym import wrappers
# Intialize P for test example
#Left =0
#Down = 1
#Right = 2
#Up= 3

P = {s : {a: [] for a in range(4)} for s in range(4)}
P[0][0] = [(0, 0, 0, False)]
P[0][1] = [(1, 2, -1, False)]
P[0][2] = [(1, 1, 0, False)]
P[0][3] = [(0, 0, 0, False)]
P[1][0] = [(1, 0, -1, False)]
P[1][1] = [(1, 3, 1, True)]
P[1][2] = [(0, 0, 0, False)]
P[1][3] = [(0, 0, 0, False)]
P[2][0] = [(0, 0, 0, False)]
P[2][1] = [(0, 0, 0, False)]
P[2][2] = [(1, 3, 1, True)]
P[2][3] = [(1, 0, 0, False)]
P[3][0] = [(0, 0, 0, True)]
P[3][1] = [(0, 0, 0, True)]
P[3][2] = [(0, 0, 0, True)]
P[3][3] = [(0, 0, 0, True)]



# Problem 1
def value_iteration(P, nS ,nA, beta = 1, tol=1e-8, maxiter=3000):
    """Perform Value Iteration according to the Bellman optimality principle.

    Parameters:
        P (dict): The Markov relationship
                (P[state][action] = [(prob, nextstate, reward, is_terminal)...]).
        nS (int): The number of states.
        nA (int): The number of actions.
        beta (float): The discount rate (between 0 and 1).
        tol (float): The stopping criteria for the value iteration.
        maxiter (int): The maximum number of iterations.

    Returns:
       v (ndarray): The discrete values for the true value function.
       n (int): number of iterations
    """
    #instantiate Value at 0
    V_old = np.zeros(nS)
    V_new = np.zeros(nS)
    sa_vector = np.zeros(nA)
    i = 1

    while i < maxiter:
        for s in range(nS):
            for a in range(nA):
                for tuple_info in P[s][a]:
                    # tuple_info is a tuple of (probability, next state, reward, done)
                    p, s_, u, _ = tuple_info
                    # sums up the possible end states and rewards with given action
                    sa_vector[a] += (p * (u + beta * V_old[s_]))
                #add the max value to the value function
                V_new[s] = np.max(sa_vector)
            sa_vector = np.zeros(nA)
        i += 1
        #check for convergence
        if la.norm(V_new - V_old) < tol:
            return V_new, i
        V_old = V_new
            

# Problem 2
def extract_policy(P, nS, nA, v, beta = 1.0):
    """Returns the optimal policy vector for value function v

    Parameters:
        P (dict): The Markov relationship
                (P[state][action] = [(prob, nextstate, reward, is_terminal)...]).
        nS (int): The number of states.
        nA (int): The number of actions.
        v (ndarray): The value function values.
        beta (float): The discount rate (between 0 and 1).

    Returns:
        policy (ndarray): which direction to move in from each square.
    """
    policy_func = np.zeros(nS)

    for s in range(nS):
        temp_max = np.zeros(nA)
        for a in range(nA):
            for tuple_info in P[s][a]:
                    # tuple_info is a tuple of (probability, next state, reward, done)
                    p, s_, u, _ = tuple_info
                    # sums up the possible end states and rewards with given action
                    temp_max[a] += (p * (u + beta * v[s_]))
        policy_func[s] = np.argmax(temp_max)

    return policy_func


# Problem 3
def compute_policy_v(P, nS, nA, policy, beta=1.0, tol=1e-8):
    """Computes the value function for a policy using policy evaluation.

    Parameters:
        P (dict): The Markov relationship
                (P[state][action] = [(prob, nextstate, reward, is_terminal)...]).
        nS (int): The number of states.
        nA (int): The number of actions.
        policy (ndarray): The policy to estimate the value function.
        beta (float): The discount rate (between 0 and 1).
        tol (float): The stopping criteria for the value iteration.

    Returns:
        v (ndarray): The discrete values for the true value function.
    """
    V_k_1 = np.zeros(nS)
    V_k = np.zeros(nS)
    while True:
        for s in range(nS):

            sa = 0
            a = int(policy[s])
            if a not in P[s]:
                V_k_1[s] = 0
            else:
                for tuple_info in P[s][a]:
                    # tuple_info is a tuple of (probability, next state, reward, done)
                    p, s_, u, _ = tuple_info
                    # sums up the possible end states and rewards with given action
                    sa += (p * (u + beta * policy[s_]))
                #add the max value to the value function
            V_k_1[s] = sa

        if la.norm(V_k_1 - V_k) < tol:
            return V_k_1
            
        V_k = V_k_1.copy()

# Problem 4
def policy_iteration(P, nS, nA, beta=1, tol=1e-8, maxiter=200):
    """Perform Policy Iteration according to the Bellman optimality principle.

    Parameters:
        P (dict): The Markov relationship
                (P[state][action] = [(prob, nextstate, reward, is_terminal)...]).
        nS (int): The number of states.
        nA (int): The number of actions.
        beta (float): The discount rate (between 0 and 1).
        tol (float): The stopping criteria for the value iteration.
        maxiter (int): The maximum number of iterations.

    Returns:
    	v (ndarray): The discrete values for the true value function
        policy (ndarray): which direction to move in each square.
        n (int): number of iterations
    """
    pi_k = np.arange(nS)
    k = 1

    while k < maxiter:

        #policy evaluation and improvement
        V_k_1 = compute_policy_v(P, nS, nA, pi_k, beta)
        pi_k_1 = extract_policy(P, nS, nA, V_k_1, beta)

        #check for convergence
        if la.norm(pi_k_1 - pi_k) < tol:
            return V_k_1, pi_k_1, k

        #reset variables
        pi_k = pi_k_1
        k += 1

    return V_k_1, pi_k_1, k


# Problem 5 and 6
def frozen_lake(basic_case=True, M=1000, render=False):
    """ Finds the optimal policy to solve the FrozenLake problem

    Parameters:
    basic_case (boolean): True for 4x4 and False for 8x8 environemtns.
    M (int): The number of times to run the simulation using problem 6.
    render (boolean): Whether to draw the environment.

    Returns:
    vi_policy (ndarray): The optimal policy for value iteration.
    vi_total_rewards (float): The mean expected value for following the value iteration optimal policy.
    pi_value_func (ndarray): The maximum value function for the optimal policy from policy iteration.
    pi_policy (ndarray): The optimal policy for policy iteration.
    pi_total_rewards (float): The mean expected value for following the policy iteration optimal policy.
    """
    if basic_case == True:
        # Make environment for 4x4 scenario
        env_name  = 'FrozenLake-v1'
    else:
        # Make environment for 8x8 scenario
        env_name = 'FrozenLake8x8-v1'


    env = gym.make(env_name).env
    # Find number of states and actions
    nS = env.observation_space.n
    nA = env.action_space.n

    # Get the dictionary with all the states and actions
    P = env.P
    value_func = value_iteration(P, nS, nA)[0]

    #get vi_policy
    vi_policy = extract_policy(P, nS, nA, value_func)
    pi_value_func, pi_policy, _ = policy_iteration(P, nS, nA)

    vi = 0
    pi = 0

    #collect mean expected rewards
    for _ in range(M):
        vi += run_simulation(env, vi_policy)
        pi += run_simulation(env, pi_policy)
    

    vi_total_rewards = vi / float(M)
    pi_total_rewards = pi / float(M)
    env.close()

    return vi_policy, vi_total_rewards, pi_value_func, pi_policy, pi_total_rewards




# Problem 6
def run_simulation(env, policy, render=True, beta = 1.0):
    """ Evaluates policy by using it to run a simulation and calculate the reward.

    Parameters:
    env (gym environment): The gym environment.
    policy (ndarray): The policy used to simulate.
    beta float: The discount factor.
    render (boolean): Whether to draw the environment.

    Returns:
    total reward (float): Value of the total reward received under policy.
    """
    #define variables
    obs = env.reset()
    done = False
    total_reward = 0
    k = 0

    #handle both cases of rendering
    if render == True:
        env.render(mode = 'human')
        obs = env.reset()

        #collect reward by incrementation
        if done != True:
            k += 1
            obs, reward, done, _ = env.step(int(policy[obs]))
            env.render(mode = 'human')
            total_reward += (beta ** k) * reward

    else:
        obs = env.reset()
        #collect reward by incrementation
        if done != True:
            k += 1
            obs, reward, done, _ = env.step(int(policy[obs]))
            total_reward += beta**k*reward

    return total_reward

if __name__ == "__main__":
    print(frozen_lake())