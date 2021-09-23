# test_specs.py
"""Python Essentials: Unit Testing.
<Sophie Gee>
<Section 3>
<09/23/21>
"""


import specs
import pytest


def test_add():
    assert specs.add(1, 3) == 4, "failed on positive integers"
    assert specs.add(-5, -7) == -12, "failed on negative integers"
    assert specs.add(-6, 14) == 8

def test_divide():
    assert specs.divide(4,2) == 2, "integer division"
    assert specs.divide(5,4) == 1.25, "float division"
    with pytest.raises(ZeroDivisionError) as excinfo:
        specs.divide(4, 0)
    assert excinfo.value.args[0] == "second input cannot be zero"


# Problem 1: write a unit test for specs.smallest_factor(), then correct it.
def test_smallest_factor():
    assert specs.smallest_factor(29) == 29, "failed on prime integers"
    assert specs.smallest_factor(1) == 1, "failed on smallest integers"
    assert specs.smallest_factor(49) == 7, "failed for square of primes"

# Problem 2: write a unit test for specs.month_length().
def test_month_length():
    assert specs.month_length("January") == 31, "failed for January"
    assert specs.month_length("February", True) == 29, "failed for leap year February"
    assert specs.month_length("February", False) == 28, "failed for non leap year February"
    assert specs.month_length("March") == 31, "failed for March"
    assert specs.month_length("April") == 30, "failed for April"
    assert specs.month_length("May") == 31, "failed for May"
    assert specs.month_length("June") == 30, "failed for June"
    assert specs.month_length("July") == 31, "failed for July"
    assert specs.month_length("August") == 31, "failed for August"
    assert specs.month_length("September") == 30, "failed for September"
    assert specs.month_length("October") == 31, "failed for October"
    assert specs.month_length("November") == 30, "failed for November"
    assert specs.month_length("December") == 31, "failed for December"
    assert specs.month_length("Deceber") == None, "failed for non-month"

# Problem 3: write a unit test for specs.operate().
def test_operate():
    assert specs.operate(4,2, '+') == 6, "failed for integer addition"
    assert specs.operate(8,6, '-') == 2, "failed for integer subtraction"
    assert specs.operate(4,2, '*') == 8, "failed for integer multiplication"
    assert specs.operate(9,3, '/') == 3, "failed for integer division"
    with pytest.raises(ZeroDivisionError) as excinfo:
        specs.operate(4, 0, '/')
    assert excinfo.value.args[0] == "division by zero is undefined"
    with pytest.raises(TypeError) as excinfo:
        specs.operate(2,3,4)
    assert excinfo.value.args[0] == "oper must be a string"
    with pytest.raises(ValueError) as excinfo:
        specs.operate(2,3,'s')
    assert excinfo.value.args[0] == "oper must be one of '+', '/', '-', or '*'"

# Problem 4: write unit tests for specs.Fraction, then correct it.
@pytest.fixture
def set_up_fractions():
    frac_1_3 = specs.Fraction(1, 3)
    frac_1_2 = specs.Fraction(1, 2)
    frac_n2_3 = specs.Fraction(-2, 3)
    return frac_1_3, frac_1_2, frac_n2_3

def test_fraction_init(set_up_fractions):
    frac_1_3, frac_1_2, frac_n2_3 = set_up_fractions
    assert frac_1_3.numer == 1
    assert frac_1_2.denom == 2
    assert frac_n2_3.numer == -2
    frac = specs.Fraction(30, 42)
    assert frac.numer == 5
    assert frac.denom == 7
    with pytest.raises(ZeroDivisionError) as excinfo:
        specs.Fraction(2,0)
    assert excinfo.value.args[0] == "denominator cannot be zero"
    with pytest.raises(TypeError) as excinfo:
        specs.Fraction(3.4,9.2)
    assert excinfo.value.args[0] == "numerator and denominator must be integers"

def test_fraction_str(set_up_fractions):
    frac_1_3, frac_1_2, frac_n2_3 = set_up_fractions
    assert str(frac_1_3) == "1/3"
    assert str(frac_1_2) == "1/2"
    assert str(frac_n2_3) == "-2/3"
    assert str(specs.Fraction(2,1)) == "2"

def test_fraction_float(set_up_fractions):
    frac_1_3, frac_1_2, frac_n2_3 = set_up_fractions
    assert float(frac_1_3) == 1 / 3.
    assert float(frac_1_2) == .5
    assert float(frac_n2_3) == -2 / 3.

def test_fraction_eq(set_up_fractions):
    frac_1_3, frac_1_2, frac_n2_3 = set_up_fractions
    assert frac_1_2 == specs.Fraction(1, 2)
    assert frac_1_3 == specs.Fraction(2, 6)
    assert frac_n2_3 == specs.Fraction(8, -12)
    assert specs.Fraction(2,4) == .5

def test_fraction_add(set_up_fractions):
    frac_1_3, frac_1_2, frac_n2_3 = set_up_fractions
    assert (frac_1_3 + frac_1_2) == specs.Fraction(5,6)
    assert (frac_1_3 + frac_n2_3) == specs.Fraction(-1,3)

def test_fraction_sub(set_up_fractions):
    frac_1_3, frac_1_2, frac_n2_3 = set_up_fractions
    assert frac_1_3 - frac_1_2 == specs.Fraction(-1,6)
    assert frac_1_3 - frac_n2_3 == specs.Fraction(1,1)
    assert frac_n2_3 - frac_1_2 == specs.Fraction(-7,6)

def test_fraction_mult(set_up_fractions):
    frac_1_3, frac_1_2, frac_n2_3 = set_up_fractions
    assert (frac_1_3 * frac_1_2) == specs.Fraction(1,6)
    assert (frac_1_3 * frac_n2_3) == specs.Fraction(-2,9)

def test_fraction_truediv(set_up_fractions):
    frac_1_3, frac_1_2, frac_n2_3 = set_up_fractions
    assert (frac_1_3 / frac_1_2) == specs.Fraction(2,3)
    assert (frac_1_3 / frac_n2_3) == specs.Fraction(-1,2)
    with pytest.raises(ZeroDivisionError) as excinfo:
        frac_1_2/specs.Fraction(0,2)
    assert excinfo.value.args[0] == "cannot divide by zero"

# Problem 5: Write test cases for Set.
@pytest.fixture
def set_up_cards(): #defines valid hand of cards containing 6 sets
    set_up_cards = ["1022", "1122", "0100", "2021",
         "0010", "2201", "2111", "0020",
         "1102", "0210", "2110", "1020"]
    return set_up_cards

def test_valid_set_hand(set_up_cards): 
    set_up_cards = set_up_cards
    assert specs.count_sets(set_up_cards) == 6
    with pytest.raises(ValueError) as excinfo:
        new_hand = set_up_cards[1:]
        specs.count_sets(new_hand)
    assert excinfo.value.args[0] == "size of hand must be 12"
    with pytest.raises(ValueError) as excinfo:
        new_hand_1 = set_up_cards
        new_hand_1[-1] = set_up_cards[0]
        specs.count_sets(new_hand_1)
    assert excinfo.value.args[0] == "not all cards unique"
    with pytest.raises(ValueError) as excinfo:
        new_hand = set_up_cards
        new_hand[-1] = "102"
        specs.count_sets(new_hand)
    assert excinfo.value.args[0] == "one or more cards does not have exactly 4 digits"
    with pytest.raises(ValueError) as excinfo:
        new_hand = set_up_cards
        new_hand[-1] = "1088"
        specs.count_sets(new_hand)
    assert excinfo.value.args[0] == "one or more cards has a character other than 0, 1, or 2"

@pytest.fixture
def set_to_test(): #defines valid hand of cards containing 6 sets
    set_to_test = ("1022", "2021", "0020")
    return set_to_test

def test_valid_is_set(set_to_test):
    set_to_test = set_to_test
    assert specs.is_set(set_to_test[0], set_to_test[1], set_to_test[2]) == True
    non_set_to_test = ("1022", "1122", "1020")
    assert specs.is_set(non_set_to_test[0], non_set_to_test[1], non_set_to_test[2]) == False
