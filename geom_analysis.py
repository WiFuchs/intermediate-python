import numpy
import argparse
import itertools
from joblib import Memory

location = './cache'
memory = Memory(location, verbose=0)


#   recommendations:
#   1) static type checking & type annotations (I use mypy, but it looks like Best Practices uses something else)
#   2) main() function and if __name__ == '__main__' so that code doesn't run when/if module is imported (extendable)
#   3) since these are all pure functions, they can be memoized into a dictionary with functools (inputs are hashable)
#   4) os import is unused
#   5) joblib can memoize functions that use numpy arrays (https://joblib.readthedocs.io/en/latest/memory.html#use-case)


# memory.cache creates a cache of function calls in the ./cache directory
# run the code twice to see when function is actually run, versus when it's value is looked up from the cache
# if the function is run, it will print "running calculate_distance"
# Now, change the print statement and run it again. What do you notice?
# Change it back to it's original form and run it one more time. What does this say about the cache?
@memory.cache
def calculate_distance(atom1_coord, atom2_coord):
    print("running calculate_distance")
    # distances isn't necessary, but could be instructional
    distances = atom1_coord - atom2_coord
    return numpy.sqrt(numpy.sum(numpy.square(distances)))


def bond_check(atom_distance, minimum_length=0, maximum_length=1.5):
    # Expression chaining may be more familiar, works the same way as it does in math.
    # Even though this is very simple, it is good to have it as a function, so that we can easily update what it means
    # to do a bond_check as the code evolves
    return minimum_length < atom_distance <= maximum_length


# Exercise: Although tempting, what bad thing could happen if we memoize the open_xyz function?
# Answer: open_xyz is an impure function because it's output does not depend only on it's input.
#         For example, we could have two completely different files named "data.xyz". We would expect open_xyz to give
#         different arrays, because they contain different information. However, if we memoize, it will give the same
#         symbols & coord arrays for all files with the same name, regardless of their contents!
def open_xyz(filename):
    xyz_file = numpy.genfromtxt(fname=filename, skip_header=2, dtype='unicode')
    symbols = xyz_file[:, 0]
    coord = (xyz_file[:, 1:])
    coord = coord.astype(numpy.float)
    return symbols, coord


def main():
    # parse the arguments
    parser = argparse.ArgumentParser(
        description="This script analyzes a user given xyz file and outputs the length of the bonds.")
    parser.add_argument("xyz_file", help="The filepath for the xyz file to analyze.")
    args = parser.parse_args()

    symbols, coord = open_xyz(args.xyz_file)
    num_atoms = len(symbols)

    print("nested for-loops: ")
    for num1 in range(0, num_atoms):
        for num2 in range(0, num_atoms):
            if num1 < num2:
                bond_length_12 = calculate_distance(coord[num1], coord[num2])
                # PEP 8 test for trueness, not specific value True
                if bond_check(bond_length_12):
                    print(F'{symbols[num1]} to {symbols[num2]} : {bond_length_12:.3f}')

    print("generator expression: ")
    # this is a good place to explain the difference between list comprehensions and generator expressions
    # list comprehensions create the list and store it in memory, generator expressions return iterator and generate
    # elements on the fly. This is key for very large lists, like you might see in computational chemistry.
    for num1, num2 in ((x, y) for x in range(0, num_atoms) for y in range(0, num_atoms)):
        if num1 < num2:
            bond_length_12 = calculate_distance(coord[num1], coord[num2])
            # PEP 8 test for trueness, not specific value True
            if bond_check(bond_length_12):
                print(F'{symbols[num1]} to {symbols[num2]} : {bond_length_12:.3f}')

    print("itertools combinations")
    # we use itertools.combinations, because the ordering of the tuples does not matter
    # we use enumerate(coord) because we care about the index in the original list, as that is how we access the symbol\
    # NOT RECOMMENDED: I don't think that this is particularly readable or sensible code, but it works
    for p1, p2 in itertools.combinations(enumerate(coord), 2):
        bond_length_12 = calculate_distance(p1[1], p2[1])
        # PEP 8 test for trueness, not specific value True
        if bond_check(bond_length_12):
            print(F'{symbols[p1[0]]} to {symbols[p2[0]]} : {bond_length_12:.3f}')



if __name__ == '__main__':
    main()
