# object_oriented.py
"""Python Essentials: Object Oriented Programming.
<Sophie Gee>
<Section 3>
<9/9/21>
"""

import math

class Backpack:
    """A Backpack object class. Has a name and a list of contents.

    Attributes:
        name (str): the name of the backpack's owner.
        contents (list): the contents of the backpack.
    """

    # Problem 1: Modify __init__() and put(), and write dump().
    def __init__(self, name, color, max_size = 5):
        """Set the name and initialize an empty list of contents.

        Parameters:
            name (str): the name of the backpack's owner.
            color (str): color of the backpack
            max_size(int): max amount of contents
        """
        self.color = color
        self.max_size = max_size
        self.name = name
        self.contents = []

    def put(self, item):
        """Add an item to the backpack's list of contents.
        
        Check that the backpack does not go over capacity. 
        If there are already max_size items or more, 
        Print “No Room!” and do not add the item to the contents list.
        """
        if len(self.contents) < self.max_size:
            self.contents.append(item)
        else:
            print("No Room!")

    def take(self, item):
        """Remove an item from the backpack's list of contents."""
        self.contents.remove(item)

    def dump(self):
        """Resets the contents of the backpack to an empty list."""
        self.contents = []
    
    # Magic Methods -----------------------------------------------------------

    # Problem 3: Write __eq__() and __str__().
    def __add__(self, other):
        """Add the number of contents of each Backpack."""
        return len(self.contents) + len(other.contents)

    def __lt__(self, other):
        """Compare two backpacks. If 'self' has fewer contents
        than 'other', return True. Otherwise, return False.
        """
        return len(self.contents) < len(other.contents)
    
    def __eq__(self, other):
        """Compare two backpacks. If 'self' has same number of contents, 
        name, and color as 'other', return True. Otherwise, return False.
        """
        return len(self.contents) == len(other.contents) and self.name == other.name and self.color == other.color

    def __str__(self):
        """Returns the string representation of an object"""

        return """
        Owner:      {name}
        Color:      {color}
        Size:       {size}
        Max Size:   {max}
        Contents:   {contents}
        """.format(name=self.name, color= self.color, 
        size=len(self.contents), max = self.max_size, contents = self.contents)

# An example of inheritance. You are not required to modify this class.
class Knapsack(Backpack):
    """A Knapsack object class. Inherits from the Backpack class.
    A knapsack is smaller than a backpack and can be tied closed.

    Attributes:
        name (str): the name of the knapsack's owner.
        color (str): the color of the knapsack.
        max_size (int): the maximum number of items that can fit inside.
        contents (list): the contents of the backpack.
        closed (bool): whether or not the knapsack is tied shut.
    """
    def __init__(self, name, color):
        """Use the Backpack constructor to initialize the name, color,
        and max_size attributes. A knapsack only holds 3 item by default.

        Parameters:
            name (str): the name of the knapsack's owner.
            color (str): the color of the knapsack.
            max_size (int): the maximum number of items that can fit inside.
        """
        Backpack.__init__(self, name, color, max_size=3)
        self.closed = True

    def put(self, item):
        """If the knapsack is untied, use the Backpack.put() method."""
        if self.closed:
            print("I'm closed!")
        else:
            Backpack.put(self, item)

    def take(self, item):
        """If the knapsack is untied, use the Backpack.take() method."""
        if self.closed:
            print("I'm closed!")
        else:
            Backpack.take(self, item)

    def weight(self):
        """Calculate the weight of the knapsack by counting the length of the
        string representations of each item in the contents list.
        """
        return sum(len(str(item)) for item in self.contents)


# Problem 2: Write a 'Jetpack' class that inherits from the 'Backpack' class.
class Jetpack(Backpack):
    def __init__(self, name, color, max_size = 2, fuel = 10):
        """Set the name and initialize an empty list of contents.

        Parameters:
            name (str): the name of the jetpack's owner.
            color (str): color of the jetkpack
            max_size(int): max amount of contents
            fuel(int): amount of fuel in jetpack
        """
        self.color = color
        self.max_size = max_size
        self.name = name
        self.fuel = fuel
        self.contents = []

    def fly(self, fuel_to_be_burned):
        """Decrements the fuel attribute by the fuel_to_be_burned amount. 
        If the user tries to burn more fuel than remains, 
        Prints “Not enough fuel!” and do not decrement the fuel"""
        if fuel_to_be_burned < self.fuel:
            self.fuel =- fuel_to_be_burned
        else: 
            print("Not enough fuel!")

    def dump(self):
        """Resets the contents of the jetpack to an empty list, empties fuel"""
        self.contents = []
        self.fuel = 0

# Problem 4: Write a 'ComplexNumber' class.
class ComplexNumber:

    def __init__(self, real, imag):
        """Constructor that accepts two numbers. 
        Store the first as self.real and the second as self.imag."""
        self.real = real
        self.imag = imag

    def __str__(self):
        """ a + bi is printed out as (a+bj) for b ≥ 0 and (a-bj)
        for b < 0."""
        if self.imag >= 0:
            return f"({self.real}+{self.imag}j)"
        else:
            return f"({self.real}-{self.imag}j)"
            
    def conjugate(self):
        """Returns the object’s complex conjugate as a new ComplexNumber object"""
        return ComplexNumber(self.real, (-1*self.imag))
    
    def __abs__(self):
        """Returns the magnitude of the complex number"""
        return math.sqrt(self.real**2+self.imag**2)

    def __eq__(self, other):
        """Returns true if two ComplexNumber objects are equal;
        having the same real and imaginary parts."""
        return self.real == other.real and self.imag == other.imag
    
    def __add__(self, other):
        """Adds real and imaginary parts component-wise"""
        return ComplexNumber((self.real+other.real), (self.imag+other.imag))
    
    def __sub__(self, other):
        """Subtracts real and imaginary parts component-wise"""
        return ComplexNumber((self.real-other.real), (self.imag-other.imag))

    def __mul__(self, other):
        """Multiplies real and imaginary parts through foiling"""
        real_part = (self.real*other.real)-(self.imag*other.imag)
        imag_part = (self.imag*other.real)+(self.real*other.imag)
        return ComplexNumber(real_part, imag_part)
    
    def __truediv__(self, other):
        """Divides complex numbers by multiplying by conjugate of other/conjugate of other"""
        real_part = ((self.real*other.real)+(self.imag*other.imag))/((other.real**2)+(other.imag**2))
        imag_part = ((self.imag*other.real)-(self.real*other.imag))/((other.real**2)+(other.imag**2))
        return ComplexNumber(real_part, imag_part)