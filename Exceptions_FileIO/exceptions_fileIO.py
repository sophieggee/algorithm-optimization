# exceptions_fileIO.py
"""Python Essentials: Exceptions and File Input/Output.
<Sophie>
<Section 3>
<09/14/21>
"""

from random import choice, uniform


# Problem 1
def arithmagic():
    """
    Takes in user input to perform a magic trick and prints the result.
    Verifies the user's input at each step and raises a
    ValueError with an informative error message if any of the following occur:

    The first number step_1 is not a 3-digit number.
    The first number's first and last digits differ by less than $2$.
    The second number step_2 is not the reverse of the first number.
    The third number step_3 is not the positive difference of the first two numbers.
    The fourth number step_4 is not the reverse of the third number.
    """

    step_1 = input(
        "Enter a 3-digit number where the first and last digits differ by 2 or more: ")
    if len(str(step_1)) != 3:
        raise ValueError("You must enter a 3-digit number.")
    if int(str(step_1)[2])-int(str(step_1)[0]) < 2:
        raise ValueError("First and last digit must differ by at least 2.")
    step_2 = input(
        "Enter the reverse of the first number, obtained by reading it backwards: ")
    if str(step_2)[::-1] != str(step_1):
        raise ValueError("This number is not the reverse of the first.")
    step_3 = input("Enter the positive difference of these numbers: ")
    if step_3 != abs(step_1-step_2):
        raise ValueError("This is not the correct value of the positive difference.")
    step_4 = input("Enter the reverse of the previous result: ")
    if str(step_4)[::-1] != str(step_3):
        raise ValueError("This number is not the reverse of the previous result.")
    print(str(step_3), "+", str(step_4), "= 1089 (ta-da!)")


# Problem 2
def random_walk(max_iters=1e12): 
    """
    If the user raises a KeyboardInterrupt by pressing ctrl+c while the 
    program is running, the function should catch the exception and 
    print "Process interrupted at iteration $i$".
    If no KeyboardInterrupt is raised, print "Process completed".

    Return walk.
    """

    walk = 0
    directions = [1, -1]
    try: 
        for i in range(int(max_iters)):
            walk += choice(directions)
    except KeyboardInterrupt:
        print("Process interrupted at iteration ", i)
    else:
        print("Process completed")
    return walk


# Problems 3 and 4: Write a 'ContentFilter' class.
    """Class for reading in file
        
    Attributes:
        filename (str): The name of the file
        contents (str): the contents of the file
        
    """


class ContentFilter(object):
    # Problem 3
    def __init__(self, filename):
        """Read from the specified file. If the filename is invalid, prompt
        the user until a valid filename is given.
        """
        while True:
            try:
                with open(filename, 'r') as myfile:
                    contents = myfile.read()
                    ContentFilter.contents = contents
                    ContentFilter.filename = filename
                    ContentFilter.totalchars = len(contents)
                    ContentFilter.alphachars = sum([s.isalpha() for s in contents])
                    ContentFilter.numchars = sum([s.isdigit() for s in contents])
                    ContentFilter.whitespacechars = sum([s.isspace() for s in contents])
                    ContentFilter.numoflines = len(myfile.readlines())
                break
            except (FileNotFoundError, TypeError, OSError):
                filename = input("Please enter a valid file name: ")

 # Problem 4 ---------------------------------------------------------------
    def check_mode(self, mode):
        """Raise a ValueError if the mode is invalid."""
        if mode != 'w' or 'x' or 'a':
            raise ValueError("Must specify meaningful mode.")

    def uniform(self, outfile, mode='w', case='upper'):
        """Write the data ot the outfile in uniform case."""
        if case == 'upper':
            with open(outfile, mode) as myfile:
                myfile.write(self.contents.upper())
        elif case == 'lower':
            with open(outfile, mode) as myfile:
                myfile.write(self.contents.lower())
        else: 
            raise ValueError("Case not correctly specified.")

    def reverse(self, outfile, mode='w', unit='line'):
        """Write the data to the outfile in reverse order."""
        if unit == 'word':
            with open(outfile, mode) as myfile:
               for i in range(len(self.contents.split('\n'))):
                    myfile.write(str(self.contents.split('\n')[i].split(' ')[::-1]))
        elif unit == 'line':
            with open(outfile, mode) as myfile:
                for i in range(len(self.contents.split('\n'))-1, 0, -1):
                    myfile.write(self.contents.split('\n')[i])
        else: 
            raise ValueError("Unit not correctly specified.")

    def transpose(self, outfile, mode='w'):
        """Write the transposed version of the data to the outfile."""
        with open(outfile, mode) as myfile:
            for i in range(len(self.contents.split(' '))):
                myfile.write(str(self.contents.split('/n')[:i]))

    def __str__(self):
        """String representation: info about the contents of the file."""
        return """
Total characters:       {totchar}
Alphabetic characters:  {alpha}
Numerical characters:   {num}
Whitespace characters:  {white}
Number of lines:        {lines}
        """.format(totchar=self.totalchars, alpha= self.alphachars, 
        num=self.numchars, white = self.whitespacechars, lines = self.numoflines)

if __name__ == "__main__":
    cf = ContentFilter("cf_example1.txt")
    cf.uniform("uniform.txt", mode='w', case="upper")
    cf.uniform("uniform.txt", mode='a', case="lower")
    cf.reverse("reverse.txt", mode='w', unit="word")
    cf.reverse("reverse.txt", mode='a', unit="line")
    cf.transpose("transpose.txt", mode='w')
    print(cf)