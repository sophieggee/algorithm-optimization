# markov_chains.py
"""Volume 2: Markov Chains.
<Sophie>
<Section 3>
<11/4/21>
"""

import numpy as np
from scipy import linalg as la
from numpy.linalg import norm


class MarkovChain:
    """A Markov chain with finitely many states.

    Attributes:
        associated transition matrix that stores the information about the possible transitions between the states in the chain
        each of the columns of the transition matrix sum to 1

    """
    # Problem 1
    def __init__(self, A, states=None):
        """Check that A is column stochastic and construct a dictionary
        mapping a state's label to its index (the row / column of A that the
        state corresponds to). Save the transition matrix, the list of state
        labels, and the label-to-index dictionary as attributes.

        Parameters:
        A ((n,n) ndarray): the column-stochastic transition matrix for a
            Markov chain with n states.
        states (list(str)): a list of n labels corresponding to the n states.
            If not provided, the labels are the indices 0, 1, ..., n-1.

        Raises:
            ValueError: if A is not square or is not column stochastic.

        Example:
            >>> MarkovChain(np.array([[.5, .8], [.5, .2]], states=["A", "B"])
        corresponds to the Markov Chain with transition matrix
                                   from A  from B
                            to A [   .5      .8   ]
                            to B [   .5      .2   ]
        and the label-to-index dictionary is {"A":0, "B":1}.
        """
        n,m = np.shape(A)

        #raise error is not column stochastic, nor square matrix--
        if not np.allclose(A.sum(axis=0), np.ones(A.shape[1])):
            raise ValueError("Not column stochastic.")
        if n != m:
            raise ValueError("Not a square matrix.")
        
        #set states to default if not specified
        if states is None:
            states = [i for i in range(n)]
        
        #create enumerated dictionary
        dict = {state: i for i, state in enumerate(states)}

        #set attributes
        self.states = states
        self.dict = dict
        self.mat = A
        self.n = n

    # Problem 2
    def transition(self, state):
        """Transition to a new state by making a random draw from the outgoing
        probabilities of the state with the specified label.

        Parameters:
            state (str): the label for the current state.

        Returns:
            (str): the label of the state to transitioned to.
        """

        #access given state by iterating through dict
        index = self.dict[state]

        #access column corresponding to state
        column = self.mat[:,index]

        #find probabilities
        prob = np.random.multinomial(1, column)
        
        #get most probable transition state
        ind = np.argmax(prob)

        #find its label given its index
        label = self.states[ind]

        return label

    # Problem 3
    def walk(self, start, N):
        """Starting at the specified state, use the transition() method to
        transition from state to state N-1 times, recording the state label at
        each step.

        Parameters:
            start (str): The starting state label.

        Returns:
            (list(str)): A list of N state labels, including start.
        """
        transition_list = [start]

        #use transition N-1 times
        for n in range(N-1):
            state = self.transition(start)
            transition_list.append(state)
            start = state
        
        #return appended list
        return transition_list

    # Problem 3
    def path(self, start, stop):
        """Beginning at the start state, transition from state to state until
        arriving at the stop state, recording the state label at each step.

        Parameters:
            start (str): The starting state label.
            stop (str): The stopping state label.

        Returns:
            (list(str)): A list of state labels from start to stop.
        """
        #set initial list to start at start state
        transition_list = [start]

        #iterate until last state indicated
        while start != stop:
            start = self.transition(start)
            transition_list.append(start)
        
        return transition_list

    # Problem 4
    def steady_state(self, tol=1e-12, maxiter=40):
        """Compute the steady state of the transition matrix A.

        Parameters:
            tol (float): The convergence tolerance.
            maxiter (int): The maximum number of iterations to compute.

        Returns:
            ((n,) ndarray): The steady state distribution vector of A.

        Raises:
            ValueError: if there is no convergence within maxiter iterations.
        """
        #generate random state distribution vector
        x_0 = np.random.uniform(0,1, size=self.n)
        x_0 = x_0/sum(x_0)

        #iterate through max number of iterations indicated
        for k in range(maxiter):
            x_k1 = self.mat@x_0
            norm_vec = x_0-x_k1

            #check 1 norm and return accordingly
            if norm(norm_vec,1) < tol:
                return x_k1
            else:
                x_0 = x_k1
        
        #raise error if needed
        raise ValueError("No convergence within maxiter iterations.")



class SentenceGenerator(MarkovChain):
    """A Markov-based simulator for natural language.

    Attributes:
        markov chain models bad english
    """
    # Problem 5
    def __init__(self, filename):
        """Read the specified file and build a transition matrix from its
        contents. You may assume that the file has one complete sentence
        written on each line.
        """
        with open(filename, 'r') as myfile:
            self.contents = myfile.read()
        
        #create a set of the words, adding start and stop
        self.words = set(self.contents.split())
        self.labels = list(self.words)
        self.labels.insert(0, '$tart')
        self.labels.append('$top')

        self.states = self.labels
        self.dict = {self.labels[i]: i for i in range(len(self.labels))}
        

        #initialize sentence
        self.sentence = self.contents.split('\n')

        #set zero matrix
        n = len(self.labels)
        A = np.zeros((n,n))
        self.n = n
        self.sentence = self.sentence[:-1]
        
        #for each sentence in the training set do: 
        # Split the sentence into a list of words.
        # Prepend "$tart" and append "$top" to the list of words.
        for sent in self.sentence:
            words = (sent.split())
            words.insert(0, "$tart")
            words.append("$top")
           
            for i in range(len(words)-1):
                currentword = words[i]
                nextword = words[i+1]
                ind = self.dict[currentword]
                nextind = self.dict[nextword]
                A[nextind,ind] += 1
        
        #make sure stop transitions to stop
        A[n-1,n-1] =1

        #normalize A
        A/=A.sum(axis=0)

        self.mat = A



    # Problem 6
    def babble(self):
        """Create a random sentence using MarkovChain.path().

        Returns:
            (str): A sentence generated with the transition matrix, not
                including the labels for the $tart and $top states.

        Example:
            >>> yoda = SentenceGenerator("yoda.txt")
            >>> print(yoda.babble())
            The dark side of loss is a path as one with you.
        """
        path = self.path("$tart","$top")
        path = path[1:-1]
        babble = " ".join(path)
        #print(np.argwhere(np.isnan(self.dict)))
        print(np.argwhere(np.isnan(self.mat)))
        print([[self.mat[i,j]for i in range(self.mat.shape[0]) if self.mat[i,j] > 1]for j in range(self.mat.shape[1])])
        return babble

if __name__ == "__main__":
    yoda = SentenceGenerator("yoda.txt")
    print(yoda.babble())
    