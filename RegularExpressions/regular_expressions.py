# regular_expressions.py
"""Volume 3: Regular Expressions.
<Sophie Gee>
<Section 2>
<2/9/22>
"""
import re

# Problem 1
def prob1():
    """Compile and return a regular expression pattern object with the
    pattern string "python".

    Returns:
        (_sre.SRE_Pattern): a compiled regular expression pattern object.
    """
    pattern = re.compile("python")
    return pattern

# Problem 2
def prob2():
    """Compile and return a regular expression pattern object that matches
    the string "^{@}(?)[%]{.}(*)[_]{&}$".

    Returns:
        (_sre.SRE_Pattern): a compiled regular expression pattern object.
    """
    return re.compile(r"\^\{@\}\(\?\)\[%\]\{\.\}\(\*\)\[_\]\{&\}\$")

# Problem 3
def prob3():
    """Compile and return a regular expression pattern object that matches
    the following strings (and no other strings).

        Book store          Mattress store          Grocery store
        Book supplier       Mattress supplier       Grocery supplier

    Returns:
        (_sre.SRE_Pattern): a compiled regular expression pattern object.
    """
    return re.compile(r"^(Book|Mattress|Grocery) (store|supplier)$")

# Problem 4
def prob4():
    """Compile and return a regular expression pattern object that matches
    any valid Python identifier.

    Returns:
        (_sre.SRE_Pattern): a compiled regular expression pattern object.
    """
    return re.compile(r"^[a-zA-Z_]+[\w]*\s*[=]*\s*([0-9\.]*|['][^']*[']|[a-zA-Z_]+[\w]*\s*)*$")

# Problem 5
def prob5(code):
    """Use regular expressions to place colons in the appropriate spots of the
    input string, representing Python code. You may assume that every possible
    colon is missing in the input string.

    Parameters:
        code (str): a string of Python code without any colons.

    Returns:
        (str): code, but with the colons inserted in the right places.
    """
    #compile a string for finding the expressions that preceed colons
    colons = re.compile(r"^(\s*((if)|(elif)|(else)|(for)|(while)|(try)|(except)|(finally)|(with)|(def))(.*)$)", re.MULTILINE)

    #replace code passed in with colons appropriately
    colon_code = colons.sub(r"\1:", code)

    return colon_code
    

# Problem 6
def prob6(filename="fake_contacts.txt"):
    """John Doe 1/1/99 (123)-456-7890 john_doe90@hopefullynotarealaddress.com"""
    """Use regular expressions to parse the data in the given file and format
    it uniformly, writing birthdays as mm/dd/yyyy and phone numbers as
    (xxx)xxx-xxxx. Construct a dictionary where the key is the name of an
    individual and the value is another dictionary containing their
    information. Each of these inner dictionaries should have the keys
    "birthday", "email", and "phone". In the case of missing data, map the key
    to None.

    Returns:
        (dict): a dictionary mapping names to a dictionary of personal info.
    """

    with open(filename, "r") as file:
        contacts = file.readlines()

    #create empty dictionary
    contact_dict = {}
    
    #get the format of all first names, middle name optional
    for contact in contacts:
        first_name = re.compile(r"^[a-zA-Z]+\s*[A-Z\.]*\s*[a-zA-Z]+")
        name = first_name.findall(contact)

        #get the format of all birthdays for keys, century optional, and reformat!
        birthday = re.compile(r"[0-9]+/[0-9]+/[0-9]+")
        birthdate = birthday.findall(contact)
        if birthdate == []:
            birthdate = None
        else:
            birthdate = birthdate[0]
            #REFORMAT BIRTHDAY
            temp = birthdate.split('/')
            if len(temp[-1]) == 2: #only returns one per line, so just get the first object
                temp[-1] = "20" + temp[-1]
            if len(temp[0]) == 1:
                temp[0] = "0" + temp[0]
            if len(temp[1]) == 1:
                temp[1] = "0" + temp[1]
            birthdate = ("/").join(temp)
        

        #get the format of the emails 
        email = re.compile(r"\S+@.+\.\S+")
        email_address = email.findall(contact)
        if email_address == []:
            email_address = None
        else:
            email_address = email_address[0]

        #get teh format of the phone numbers
        phone = re.compile(r"\(?[0-9]{1,3}\)?-?[0-9]{1,3}-?[0-9]{3,4}-?[0-9]{0,4}")
        phone_number = phone.findall(contact)
        if phone_number == []:
            phone_number = None
        else:
            #REFORMAT phone number
            phone_number = phone_number[0]
            if phone_number[0] != '(':
                temp = phone_number.split('-')
                if len(temp) > 3:
                    temp.pop(0)
                phone_number = "(" + temp[0] + ")" + temp[1] + "-" + temp[2]
            elif phone_number[0] == '(' and phone_number[5] == '-':
                phone_number = phone_number[:5] +  phone_number[6:]
        
        #COMPILE AND RETURN DICTIONARY
        contact_dict[name[0]] = {
            "birthday": birthdate, 
            "email": email_address, 
            "phone": phone_number}
    return contact_dict

if __name__ == "__main__":
    print(prob6())