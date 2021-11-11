# shell2.py
"""Volume 3: Unix Shell 2.
<Sophie>
<Section 3>
<11/11/21>
"""
import os
from glob import glob
import subprocess

# Problem 3
def grep(target_string, file_pattern):
    """Find all files in the current directory or its subdirectories that
    match the file pattern, then determine which ones contain the target
    string.

    Parameters:
        target_string (str): A string to search for in the files whose names
            match the file_pattern.
        file_pattern (str): Specifies which files to search.
    """
    #get names of files with same file_pattern
    files_to_return = []

    #create the right format for glob
    file_pattern = "**/" + file_pattern
    files = glob(file_pattern, recursive=True)

    #search through files for target_string
    for file in files:
        with open(file) as myfile:
            contents = myfile.read()
        #check to see whta index .find finds 
        if contents.find(target_string) != -1:
            files_to_return.append(file)
    return files_to_return

# Problem 4
def largest_files(n):
    """Return a list of the n largest files in the current directory or its
    subdirectories (from largest to smallest).
    """
    file_info = []
    for directory, subdirectories, files in os.walk('.'):
        for filename in files:
            filename = os.path.join(directory,filename)
            file_size = subprocess.check_output(["wc", "-c", filename]).decode()
            file_info.append(tuple(file_size.split()))
    #sort file sizes
    file_info.sort(key=lambda x: int(x[0]), reverse = True)
    largest_n = file_info[:n]
    
    #write smallest n to smallest.txt file
    with open("smallest.txt", "w") as f:
        f.write(largest_n[-1][0])
    return [i[1] for i in largest_n]
    
# Problem 6    
def prob6(n = 10):
   """this problem counts to or from n three different ways, and
      returns the resulting lists each integer
   
   Parameters:
       n (int): the integer to count to and down from
   Returns:
       integerCounter (list): list of integers from 0 to the number n
       twoCounter (list): list of integers created by counting down from n by two
       threeCounter (list): list of integers created by counting up to n by 3
   """
   #print what the program is doing
   integerCounter = list()
   twoCounter = list()
   threeCounter = list()
   counter = n
   for i in range(n+1):
       integerCounter.append(i)
       if (i % 2 == 0):
           twoCounter.append(counter - i)
       if (i % 3 == 0):
           threeCounter.append(i)
   #return relevant values
   return integerCounter, twoCounter, threeCounter

if __name__ == "__main__":
    print(largest_files(3))