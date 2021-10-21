# binary_trees.py
"""Volume 2: Binary Trees.
<Sophie Gee>
<section3>
<oct 6>
"""

# These imports are used in BST.draw().
import networkx as nx
import numpy as np
import random
import time
from networkx.algorithms.dag import root_to_leaf_paths
from networkx.drawing.nx_agraph import graphviz_layout
from matplotlib import pyplot as plt

class SinglyLinkedListNode:
    """A node with a value and a reference to the next node."""
    def __init__(self, data):
        self.value, self.next = data, None

class SinglyLinkedList:
    """A singly linked list with a head and a tail."""
    def __init__(self):
        self.head, self.tail = None, None

    def append(self, data):
        """Add a node containing the data to the end of the list."""
        n = SinglyLinkedListNode(data)
        if self.head is None:
            self.head, self.tail = n, n
        else:
            self.tail.next = n
            self.tail = n

    def iterative_find(self, data):
        """Search iteratively for a node containing the data.
        If there is no such node in the list, including if the list is empty,
        raise a ValueError.

        Returns:
            (SinglyLinkedListNode): the node containing the data.
        """
        current = self.head
        while current is not None:
            if current.value == data:
                return current
            current = current.next
        raise ValueError(str(data) + " is not in the list")

    # Problem 1
    def recursive_find(self, data):
        """Search recursively for the node containing the data.
        If there is no such node in the list, including if the list is empty,
        raise a ValueError.

        Returns:
            (SinglyLinkedListNode): the node containing the data.
        """
        def check_node(n): #defines inner function
            if n is None: #checks if node is none
                raise ValueError("The data could not be found.")
            if n.value == data: #checks if node is data
                return n
            return check_node(n.next) #calls on next node
        return check_node(self.head)


class BSTNode:
    """A node class for binary search trees. Contains a value, a
    reference to the parent node, and references to two child nodes.
    """
    def __init__(self, data):
        """Construct a new node and set the value attribute. The other
        attributes will be set when the node is added to a tree.
        """
        self.value = data
        self.prev = None        # A reference to this node's parent node.
        self.left = None        # self.left.value < self.value
        self.right = None       # self.value < self.right.value


class BST:
    """Binary search tree data structure class.
    The root attribute references the first node in the tree.
    """
    def __init__(self):
        """Initialize the root attribute."""
        self.root = None

    def find(self, data):
        """Return the node containing the data. If there is no such node
        in the tree, including if the tree is empty, raise a ValueError.
        """

        # Define a recursive function to traverse the tree.
        def _step(current):
            """Recursively step through the tree until the node containing
            the data is found. If there is no such node, raise a Value Error.
            """
            if current is None:                     # Base case 1: dead end.
                raise ValueError(str(data) + " is not in the tree.")
            if data == current.value:               # Base case 2: data found!
                return current
            if data < current.value:                # Recursively search left.
                return _step(current.left)
            else:                                   # Recursively search right.
                return _step(current.right)

        # Start the recursion on the root of the tree.
        return _step(self.root)

    # Problem 2
    def insert(self, data):
        """Insert a new node containing the specified data.

        Raises:
            ValueError: if the data is already in the tree.

        Example:
            >>> tree = BST()                    |
            >>> for i in [4, 3, 6, 5, 7, 8, 1]: |            (4)
            ...     tree.insert(i)              |            / \
            ...                                 |          (3) (6)
            >>> print(tree)                     |          /   / \
            [4]                                 |        (1) (5) (7)
            [3, 6]                              |                  \
            [1, 5, 7]                           |                  (8)
            [8]                                 |
        """
        
        def _step(current,parent):
            """Recursively step through the tree until the node containing
            the data is found. If there is no such node, raise a Value Error.
            """
            if current is None:                     # Base case 1: dead end.
                return parent
            if data == current.value:               # Base case 2: data found!
                raise ValueError("Data already in tree")
            if data < current.value: # Recursively search left.
                return _step(current.left,current)
            else:                                   # Recursively search right.
                return _step(current.right, current)
            
        if self.root is None:
            self.root = BSTNode(data)
        else:
            new_node = BSTNode(data)
            parent = _step(self.root, None)
            new_node.prev = parent
            if parent.value > data:
                parent.left = new_node
            else:
                parent.right = new_node

    # Problem 3
    def remove(self, data):
        """Remove the node containing the specified data.

        Raises:
            ValueError: if there is no node containing the data, including if
                the tree is empty.

        Examples:
            >>> print(12)                       | >>> print(t3)
            [6]                                 | [5]
            [4, 8]                              | [3, 6]
            [1, 5, 7, 10]                       | [1, 4, 7]
            [3, 9]                              | [8]
            >>> for x in [7, 10, 1, 4, 3]:      | >>> for x in [8, 6, 3, 5]:
            ...     t1.remove(x)                | ...     t3.remove(x)
            ...                                 | ...
            >>> print(t1)                       | >>> print(t3)
            [6]                                 | [4]
            [5, 8]                              | [1, 7]
            [9]                                 |
                                                | >>> print(t4)
            >>> print(t2)                       | [5]
            [2]                                 | >>> t4.remove(1)
            [1, 3]                              | ValueError: <message>
            >>> for x in [2, 1, 3]:             | >>> t4.remove(5)
            ...     t2.remove(x)                | >>> print(t4)
            ...                                 | []
            >>> print(t2)                       | >>> t4.remove(5)
            []                                  | ValueError: <message>
        """
        node = BST.find(self,data)
        def minValueNode(node):
            current = node
            # loop down to find the leftmost leaf of this node
            while(current.right is not None):
                current = current.right
 
            return current #returns the left-most leaf of given node
        node_to_delete = node.value
        def deleteNode(root, key):
            if root is None:
                return root #base case
            if key < root.value: #if node value is less than root value then put on left
                root.left = deleteNode(root.left, key)
            elif key > root.value: #if node value is greater than root value, put on right
                root.right = deleteNode(root.right, key)
            else: #keep traversing down the tree 
                if root.left is None: #to find if has no children,
                    k = root.right
                    print("here line 206")
                    root = None
                    print("here line 207")
                    if key == self.root.value:
                        self.root=self.root.right
                        print("here line 209")
                    return root
                elif root.right is None:
                    k = root.left
                    if root.prev.right is root:
                        root.prev.right = k
                        k.prev = root.prev
                        root.left = None
                        #print(k.prev.right.value)
                        #root = None
                        
                    elif root.prev.left is root:
                        root.prev.left = k
                        k.prev = root.prev
                        root.prev.left = None
                        #root = None
                    #print(k.value, root.prev.left.value)
                    if key == self.root.value:
                        self.root=self.root.left
                        print("here left")

                    return root
                k = minValueNode(root.left) #find the predecessor and store to replace
                if key == self.root.value:
                    self.root.value = k.value
                    print("here line 225")
                    if k.prev.right.value == k.value:
                        k.prev.right = None
                        print("here line 225")
                    else:
                        k.prev.left = None
                        print("here line 228")
                    return k
                root.value = k.value
                print("here line 234")
                root.left = deleteNode(root.left, k.value)
            return root
        deleteNode(self.root, node_to_delete) #recursive call start


    def __str__(self):
        """String representation: a hierarchical view of the BST.

        Example:  (3)
                  / \     '[3]          The nodes of the BST are printed
                (2) (5)    [2, 5]       by depth levels. Edges and empty
                /   / \    [1, 4, 6]'   nodes are not printed.
              (1) (4) (6)
        """
        if self.root is None:                       # Empty tree
            return "[]"
        out, current_level = [], [self.root]        # Nonempty tree
        while current_level:
            next_level, values = [], []
            for node in current_level:
                values.append(node.value)
                for child in [node.left, node.right]:
                    if child is not None:
                        next_level.append(child)
            out.append(values)
            current_level = next_level
        return "\n".join([str(x) for x in out])

    def draw(self):
        """Use NetworkX and Matplotlib to visualize the tree."""
        if self.root is None:
            return

        # Build the directed graph.
        G = nx.DiGraph()
        G.add_node(self.root.value)
        nodes = [self.root]
        while nodes:
            current = nodes.pop(0)
            for child in [current.left, current.right]:
                if child is not None:
                    G.add_edge(current.value, child.value)
                    nodes.append(child)

        # Plot the graph. This requires graphviz_layout (pygraphviz).
        nx.draw(G, pos=graphviz_layout(G, prog="dot"), arrows=True,
                with_labels=True, node_color="C1", font_size=8)
        plt.show()


class AVL(BST):
    """Adelson-Velsky Landis binary search tree data structure class.
    Rebalances after insertion when needed.
    """
    def insert(self, data):
        """Insert a node containing the data into the tree, then rebalance."""
        BST.insert(self, data)      # Insert the data like usual.
        n = self.find(data)
        while n:                    # Rebalance from the bottom up.
            n = self._rebalance(n).prev

    def remove(*args, **kwargs):
        """Disable remove() to keep the tree in balance."""
        raise NotImplementedError("remove() is disabled for this class")

    def _rebalance(self,n):
        """Rebalance the subtree starting at the specified node."""
        balance = AVL._balance_factor(n)
        if balance == -2:                                   # Left heavy
            if AVL._height(n.left.left) > AVL._height(n.left.right):
                n = self._rotate_left_left(n)                   # Left Left
            else:
                n = self._rotate_left_right(n)                  # Left Right
        elif balance == 2:                                  # Right heavy
            if AVL._height(n.right.right) > AVL._height(n.right.left):
                n = self._rotate_right_right(n)                 # Right Right
            else:
                n = self._rotate_right_left(n)                  # Right Left
        return n

    @staticmethod
    def _height(current):
        """Calculate the height of a given node by descending recursively until
        there are no further child nodes. Return the number of children in the
        longest chain down.
                                    node | height
        Example:  (c)                  a | 0
                  / \                  b | 1
                (b) (f)                c | 3
                /   / \                d | 1
              (a) (d) (g)              e | 0
                    \                  f | 2
                    (e)                g | 0
        """
        if current is None:     # Base case: the end of a branch.
            return -1           # Otherwise, descend down both branches.
        return 1 + max(AVL._height(current.right), AVL._height(current.left))

    @staticmethod
    def _balance_factor(n):
        return AVL._height(n.right) - AVL._height(n.left)

    def _rotate_left_left(self, n):
        temp = n.left
        n.left = temp.right
        if temp.right:
            temp.right.prev = n
        temp.right = n
        temp.prev = n.prev
        n.prev = temp
        if temp.prev:
            if temp.prev.value > temp.value:
                temp.prev.left = temp
            else:
                temp.prev.right = temp
        if n is self.root:
            self.root = temp
        return temp

    def _rotate_right_right(self, n):
        temp = n.right
        n.right = temp.left
        if temp.left:
            temp.left.prev = n
        temp.left = n
        temp.prev = n.prev
        n.prev = temp
        if temp.prev:
            if temp.prev.value > temp.value:
                temp.prev.left = temp
            else:
                temp.prev.right = temp
        if n is self.root:
            self.root = temp
        return temp

    def _rotate_left_right(self, n):
        temp1 = n.left
        temp2 = temp1.right
        temp1.right = temp2.left
        if temp2.left:
            temp2.left.prev = temp1
        temp2.prev = n
        temp2.left = temp1
        temp1.prev = temp2
        n.left = temp2
        return self._rotate_left_left(n)

    def _rotate_right_left(self, n):
        temp1 = n.right
        temp2 = temp1.left
        temp1.left = temp2.right
        if temp2.right:
            temp2.right.prev = temp1
        temp2.prev = n
        temp2.right = temp1
        temp1.prev = temp2
        n.right = temp2
        return self._rotate_right_right(n)


# Problem 4
def prob4():
    """Compare the build and search times of the SinglyLinkedList, BST, and
    AVL classes. For search times, use SinglyLinkedList.iterative_find(),
    BST.find(), and AVL.find() to search for 5 random elements in each
    structure. Plot the number of elements in the structure versus the build
    and search times. Use log scales where appropriate.
    """
    with open("english.txt", "r") as myfile: #open my english file and create list of the lines
        list_of_words = myfile.readlines() 
    domain = 2**np.arange(3,11)
    times = []
    times2 = []
    times3 = []
    
    for n in range(3,11): #iterate through at differing sizes of n
        single = SinglyLinkedList() #initialize structures
        bst = BST()
        avl = AVL()
        subset = random.sample(list_of_words, 2**n)
        start = time.time()
        for i in subset: #append on list, bst, and avl
            single.append(i)
        times.append(time.time()-start)
        start1 = time.time()
        for i in subset:
            bst.insert(i)
        times2.append(time.time()-start1)
        start2 = time.time()
        for i in subset:
            avl.insert(i)
        times3.append(time.time()-start2)
    plt.subplot(121) #plot first subplot
    plt.loglog(domain, times, '.-', linewidth=2, markersize=15, label="SinglyLinkedList")
    plt.loglog(domain, times2, '.-', linewidth=2, markersize=15, label="Binary Search Tree")
    plt.loglog(domain, times3, '.-', linewidth=2, markersize=15, label="AVL")
    plt.ylabel("Time taken to build each structure given lists of size x")
    plt.xlabel("Lists size")
    plt.title("Time Taken to Build")
    plt.legend()


    domain = 2**np.arange(3,11)
    times = []
    times2 = []
    times3 = []
    
    for n in range(3,11): #iterate through differing n sizes
        single = SinglyLinkedList() #initialize data structures
        bst = BST()
        avl = AVL()
        subset = random.sample(list_of_words, 2**n)
        random_five = random.sample(subset, 5)
        for i in subset: #populate data structures
            single.append(i)
            avl.insert(i)
            bst.insert(i)
        start = time.time()
        for i in random_five: #find elements in each data structure
            single.iterative_find(i)
        times.append(time.time()-start)
        start1 = time.time()
        for i in random_five:
            bst.find(i)
        times2.append(time.time()-start1)
        start2 = time.time()
        for i in random_five:
            avl.find(i)
        times3.append(time.time()-start2)

    plt.subplot(122) #plot the second subplot
    plt.loglog(domain, times, '.-', linewidth=2, markersize=15, label="SinglyLinkedList")
    plt.loglog(domain, times2, '.-', linewidth=2, markersize=15, label="Binary Search Tree")
    plt.loglog(domain, times3, '.-', linewidth=2, markersize=15, label="AVL")
    plt.ylabel("time taken to find 5 from each list of size x")
    plt.xlabel("Lists size")
    plt.title("Time Taken to Find")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    nodes = [4, 2, 10, 1, 3, 5, 11, 6, 15, 9, 14, 16, 7, 12]

    bst = BST()

    for node in nodes:
        bst.insert(node)

    print(bst)
    bst.remove(9)
    print()
    print(bst)

 