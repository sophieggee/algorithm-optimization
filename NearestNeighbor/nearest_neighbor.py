# nearest_neighbor.py
"""Volume 2: Nearest Neighbor Search.
<Sophie Gee>
<section 3>
<10/21/21>
"""

import numpy as np
from scipy import linalg as la
from scipy import stats


# Problem 1
def exhaustive_search(X, z):
    """Solve the nearest neighbor search problem with an exhaustive search.

    Parameters:
        X ((m,k) ndarray): a training set of m k-dimensional points.
        z ((k, ) ndarray): a k-dimensional target point.

    Returns:
        ((k,) ndarray) the element (row) of X that is nearest to z.
        (float) The Euclidean distance from the nearest neighbor to z.
    """
    norms = la.norm(X-z, axis=1)
    return X[np.argmin(norms)], min(norms)



# Problem 2: Write a KDTNode class.
class KDTNode:
    """Node class for K-D Trees.

    Attributes:
        left (KDTNode): a reference to this node's left child.
        right (KDTNode): a reference to this node's right child.
        value ((k,) ndarray): a coordinate in k-dimensional space.
        pivot (int): the dimension of the value to make comparisons on.
    """
    def __init__(self,x):
        if type(x) is not np.ndarray:
            raise ValueError("X must be type ndarray.")
        else:
            self.value = x
            self.left = None
            self.right = None
            self.pivot = None


# Problems 3 and 4
class KDT:
    """A k-dimensional binary tree for solving the nearest neighbor problem.

    Attributes:
        root (KDTNode): the root node of the tree. Like all other nodes in
            the tree, the root has a NumPy array of shape (k,) as its value.
        k (int): the dimension of the data in the tree.
    """
    def __init__(self):
        """Initialize the root and k attributes."""
        self.root = None
        self.k = None

    def find(self, data):
        """Return the node containing the data. If there is no such node in
        the tree, or if the tree is empty, raise a ValueError.
        """
        def _step(current):
            """Recursively step through the tree until finding the node
            containing the data. If there is no such node, raise a ValueError.
            """
            if current is None:                     # Base case 1: dead end.
                raise ValueError(str(data) + " is not in the tree")
            elif np.allclose(data, current.value):
                return current                      # Base case 2: data found!
            elif data[current.pivot] < current.value[current.pivot]:
                return _step(current.left)          # Recursively search left.
            else:
                return _step(current.right)         # Recursively search right.

        # Start the recursive search at the root of the tree.
        return _step(self.root)

    # Problem 3
    def insert(self, data):
        """Insert a new node containing the specified data.

        Parameters:
            data ((k,) ndarray): a k-dimensional point to insert into the tree.

        Raises:
            ValueError: if data does not have the same dimensions as other
                values in the tree.
            ValueError: if data is already in the tree
        """
        def _step(current, previous):
                """Recursively step through the tree until finding the node
                containing the data of the insertion's parent. If there is no 
                such node, raise a ValueError.
                """
                if current is None:                     # Base case 1: dead end.
                    node = KDTNode(data)
                    if previous.pivot == self.k-1:
                        node.pivot = 0
                    else:
                        node.pivot = previous.pivot+1
                    if data[previous.pivot] < previous.value[previous.pivot]:
                        previous.left = node
                    else:
                        previous.right = node    
                elif data[current.pivot] < current.value[current.pivot]:
                    return _step(current.left, current)          # Recursively search left.
                else:
                    return _step(current.right, current)
            
        if self.root == None:
            node = KDTNode(data) 
            node.pivot = 0
            self.root = node
            self.k = len(data)
        elif len(data) != self.k:
            raise ValueError(f"cannot insert data of length {len(data)} in this tree.")
        else: 
            try:
                self.find(data)
            except ValueError:
                #Insert data that we want to insert
                _step(self.root, None)
            else:
                raise ValueError("Node trying to insert is already in kdtree.")
        

    # Problem 4
    def query(self, z):
        """Find the value in the tree that is nearest to z.

        Parameters:
            z ((k,) ndarray): a k-dimensional target point.

        Returns:
            ((k,) ndarray) the value in the tree that is nearest to z.
            (float) The Euclidean distance from the nearest neighbor to z.
        """
        def KDSearch(current, nearest, d):
            if current is None:
                return nearest, d
            x = current.value
            i = current.pivot
            if la.norm(x-z) < d:
                nearest = current
                d = la.norm(x-z)
            if z[i] < x[i]:
                nearest, d = KDSearch(current.left, nearest, d)
                if z[i]+d >= x[i]:
                    nearest, d = KDSearch(current.right, nearest, d)
                
            else:
                nearest, d = KDSearch(current.right, nearest, d)
                if (z[i] - d) <= x[i]:
                    nearest, d = KDSearch(current.left, nearest, d)
            return nearest, d
        node, d = KDSearch(self.root, self.root, la.norm(self.root.value-z))
        return node.value, d

    def __str__(self):
        """String representation: a hierarchical list of nodes and their axes.

        Example:                           'KDT(k=2)
                    [5,5]                   [5 5]   pivot = 0
                    /   \                   [3 2]   pivot = 1
                [3,2]   [8,4]               [8 4]   pivot = 1
                    \       \               [2 6]   pivot = 0
                    [2,6]   [7,5]           [7 5]   pivot = 0'
        """
        if self.root is None:
            return "Empty KDT"
        nodes, strs = [self.root], []
        while nodes:
            current = nodes.pop(0)
            strs.append("{}\tpivot = {}".format(current.value, current.pivot))
            for child in [current.left, current.right]:
                if child:
                    nodes.append(child)
        return "KDT(k={})\n".format(self.k) + "\n".join(strs)


# Problem 5: Write a KNeighborsClassifier class.
class KNeighborsClassifier:
    """A k-nearest neighbors classifier that uses SciPy's KDTree to solve
    the nearest neighbor problem efficiently.
    """
    def __init__(self, n_neighbors):
        """Initialize the n_neighbors and k attributes."""
        self.neighbors = n_neighbors
        
    
    def fit(self,X, y):
        self.tree = KDTree(X)
        self.labels = (y)

    def predict(self, z):
        min_distance, index = self.tree.query(z, k= self.neighbors)
        labels = self.labels[index]
        return stats.mode(labels)[0]
# Problem 6
def prob6(n_neighbors, filename="mnist_subset.npz"):
    """Extract the data from the given file. Load a KNeighborsClassifier with
    the training data and the corresponding labels. Use the classifier to
    predict labels for the test data. Return the classification accuracy, the
    percentage of predictions that match the test labels.

    Parameters:
        n_neighbors (int): the number of neighbors to use for classification.
        filename (str): the name of the data file. Should be an npz file with
            keys 'X_train', 'y_train', 'X_test', and 'y_test'.

    Returns:
        (float): the classification accuracy.
    """
    data = np.load("mnist_subset.npz")
    X_train = data["X_train"].astype(np.float)
    y_train = data["y_train"]
    X_test = data["X_test"].astype(np.float)
    y_test = data["y_test"]
    classifier = KNeighborsClassifier(n_neighbors)
    classifier.fit(X_train, y_train).predict(X_test)

if __name__ == "__main__":
    from scipy.spatial import KDTree
# Initialize the tree with data (in this example, use random data).
    data = np.random.random((100,5))    # 100 5-dimensional points.
    target = np.random.random(5)
    tree = KDTree(data)
# Query the tree for the nearest neighbor and its distance from 'target'.
    min_distance, index = tree.query(target)
    print(min_distance)
    print(tree.data[index])
    tree2 = KDT()
    for i in data:
        tree2.insert(i)
    # Query the tree for the nearest neighbor and its distance from 'target'.
    min_distance, index = tree2.query(target)
    print(min_distance)
    print(index)

    #print(tree)
