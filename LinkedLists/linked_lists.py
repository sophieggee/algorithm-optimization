# linked_lists.py
"""Volume 2: Linked Lists.
<Sophie Gee>
<section 3>
<09/29/21>
"""


# Problem 1
class Node:
    """A basic node class for storing data of type int, float, or str."""
    def __init__(self, data):
        """Store the data in the value attribute.
                
        Raises:
            TypeError: if data is not of type int, float, or str.
        """
        if isinstance(data, (int, float, str)):
            self.value = data
        else:
            raise TypeError("Data type must be int, float, or string.")


class LinkedListNode(Node):
    """A node class for doubly linked lists. Inherits from the Node class.
    Contains references to the next and previous nodes in the linked list.
    """
    def __init__(self, data):
        """Store the data in the value attribute and initialize
        attributes for the next and previous nodes in the list.
        """
        Node.__init__(self, data)       # Use inheritance to set self.value.
        self.next = None                # Reference to the next node.
        self.prev = None                # Reference to the previous node.


# Problems 2-5
class LinkedList:
    """Doubly linked list data structure class.

    Attributes:
        head (LinkedListNode): the first node in the list.
        tail (LinkedListNode): the last node in the list.
    """
    def __init__(self):
        """Initialize the head and tail attributes by setting
        them to None, since the list is empty initially.
        """
        self.head = None
        self.tail = None
        self.size = 0

    def append(self, data):
        """Append a new node containing the data to the end of the list."""
        # Create a new node to store the input data.
        new_node = LinkedListNode(data)
        if self.head is None:
            # If the list is empty, assign the head and tail attributes to
            # new_node, since it becomes the first and last node in the list.
            self.head = new_node
            self.tail = new_node
        else:
            # If the list is not empty, place new_node after the tail.
            self.tail.next = new_node               # tail --> new_node
            new_node.prev = self.tail               # tail <-- new_node
            # Now the last node in the list is new_node, so reassign the tail.
            self.tail = new_node
        self.size+=1

    # Problem 2
    def find(self, data):
        """Return the first node in the list containing the data.

        Raises:
            ValueError: if the list does not contain the data.

        Examples:
            >>> l = LinkedList()
            >>> for x in ['a', 'b', 'c', 'd', 'e']:
            ...     l.append(x)
            ...
            >>> node = l.find('b')
            >>> node.value
            'b'
            >>> l.find('f')
            ValueError: <message>
        """
        i = self.head
        while i != None:
            if i.value == data:
                return i
            i = i.next
        else: 
            raise ValueError(f"'{data}' not found in linkedlist.")

    # Problem 2
    def get(self, i):
        """Return the i-th node in the list.

        Raises:
            IndexError: if i is negative or greater than or equal to the
                current number of nodes.

        Examples:
            >>> l = LinkedList()
            >>> for x in ['a', 'b', 'c', 'd', 'e']:
            ...     l.append(x)
            ...
            >>> node = l.get(3)
            >>> node.value
            'd'
            >>> l.get(5)
            IndexError: <message>
        """
        if i < 0 or i > self.size: #checks for edge cases before greabbing an element
            raise IndexError("Index must be positive and less than size of linkedlist.")
        else:
            n = self.head
            for k in range(i): #iterates through i times
                n = n.next
            return n

    # Problem 3
    def __len__(self):
        """Return the number of nodes in the list.

        Examples:
            >>> l = LinkedList()
            >>> for i in (1, 3, 5):
            ...     l.append(i)
            ...
            >>> len(l)
            3
            >>> l.append(7)
            >>> len(l)
            4
        """
        return self.size #storing this attribute previously

    # Problem 3
    def __str__(self):
        """String representation: the same as a standard Python list.

        Examples:
            >>> l1 = LinkedList()       |   >>> l2 = LinkedList()
            >>> for i in [1,3,5]:       |   >>> for i in ['a','b',"c"]:
            ...     l1.append(i)        |   ...     l2.append(i)
            ...                         |   ...
            >>> print(l1)               |   >>> print(l2)
            [1, 3, 5]                   |   ['a', 'b', 'c']
        """
        n = self.head #assigns node to the head node
        values = []
        while True:
            if n is None:
                return str(values) #creates string of values throughout list
            values.append(n.value)
            n = n.next

    # Problem 4
    def remove(self, data):
        """Remove the first node in the list containing the data.

        Raises:
            ValueError: if the list is empty or does not contain the data.

        Examples:
            >>> print(l1)               |   >>> print(l2)
            ['a', 'e', 'i', 'o', 'u']   |   [2, 4, 6, 8]
            >>> l1.remove('i')          |   >>> l2.remove(10)
            >>> l1.remove('a')          |   ValueError: <message>
            >>> l1.remove('u')          |   >>> l3 = LinkedList()
            >>> print(l1)               |   >>> l3.remove(10)
            ['e', 'o']                  |   ValueError: <message>
        """
        target = self.find(data) #tries to find target node in list using data
        if target.prev is None: #if target does not have predecessor
            self.head = target.next
        else:
            target.prev.next = target.next
        if target.next is None: #if target does not have successor
            self.tail = target.prev
        else: 
            target.next.prev = target.prev
        self.size -= 1 #decriment size


    # Problem 5
    def insert(self, index, data):
        """Insert a node containing data into the list immediately before the
        node at the index-th location.

        Raises:
            IndexError: if index is negative or strictly greater than the
                current number of nodes.

        Examples:
            >>> print(l1)               |   >>> len(l2)
            ['b']                       |   5
            >>> l1.insert(0, 'a')       |   >>> l2.insert(6, 'z')
            >>> print(l1)               |   IndexError: <message>
            ['a', 'b']                  |
            >>> l1.insert(2, 'd')       |   >>> l3 = LinkedList()
            >>> print(l1)               |   >>> l3.insert(1, 'a')
            ['a', 'b', 'd']             |   IndexError: <message>
            >>> l1.insert(2, 'c')       |
            >>> print(l1)               |
            ['a', 'b', 'c', 'd']        |
        """
        if index == self.size: #if they want to insert at end of list
            self.append(data)
        if index > self.size or index < 0: #checks for edge cases
            raise IndexError("invalid Index")
        elif self.head is None: #if list is empty
            self.append(data)
        else:
            new_node = LinkedListNode(data) #creates node to add
            current_node = self.get(index)
            if current_node is None:
                self.head = new_node
            else:
                current_node.prev.next = new_node #attaches pointers of neighbors to node
                new_node.prev = current_node.prev
            current_node.prev = new_node
            new_node.next = current_node

# Problem 6: Deque class.
class Deque(LinkedList):
    def __init__(self): #creates deque class
        LinkedList.__init__(self)

    def pop(self): #defining pop
        if self.head is None: #if deque is empty
            raise ValueError("Deque is empty")
        node = self.tail
        if node.prev is not None: #if node has no predecessor 
            node.prev.next = None
            self.tail = node.prev
        else:
            self.head = None #empty deque
            self.tail = None        
        return node.value #returns tail
            
    def popleft(self): #returns beginning of deque
        node = self.head
        if node is None:
            raise ValueError("Deque is empty")
        LinkedList.remove(self, node.value)
        return node.value

    def appendleft(self,data): #appends at beginning of deque
        LinkedList.insert(self, 0, data)
    
    def remove(*args, **kwargs): #disarms remove
        raise NotImplementedError("Use pop() or popleft() for removal")

    def insert(*args, **kwargs): #disarms insert
        raise NotImplementedError("Use append() or appendleft() for insertion")

# Problem 7
def prob7(infile, outfile):
    """Reverse the contents of a file by line and write the results to
    another file.

    Parameters:
        infile (str): the file to read from.
        outfile (str): the file to write to.
    """
    with open(infile, "r") as myfile:
        list_of_words = myfile.readlines() 
    with open(outfile, "w") as myfile:
        while len(list_of_words) > 0:
            myfile.write(list_of_words.pop())

if __name__ == "__main__":
    prob7("english.txt","words.txt")