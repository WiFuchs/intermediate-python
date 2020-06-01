import numpy
import argparse
import itertools
from joblib import Memory

# this line tells joblib where to store the function call cache
location = './cache'
memory = Memory(location, verbose=0)


#   recommendations:
#   1) static type checking & type annotations (I use mypy, but it looks like Best Practices uses something else)
#   2) main() function and if __name__ == '__main__' so that code doesn't run when/if module is imported (extendable)
#   3) since these are all pure functions, they can be memoized into a dictionary with functools (inputs are hashable)
#   4) os import is unused
#   5) joblib can memoize functions that use numpy arrays (https://joblib.readthedocs.io/en/latest/memory.html#use-case)
#   6) Dictionary is less fragile than paired arrays


# memory.cache creates a cache of function calls in the ./cache directory
# run the code twice to see when function is actually run, versus when it's value is looked up from the cache
# if the function is run, it will print "running calculate_distance"
# Now, change the print statement and run it again. What do you notice?
# Change it back to it's original form and run it one more time. What does this say about the cache?
@memory.cache
def calculate_distance(atom1_coord, atom2_coord):
    # print in there so that we can see when this is run and when it is cached
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
    file_dict = {}
    for l in xyz_file:
        file_dict[l[0]] = l[1:].astype(numpy.float)
    return file_dict
    # equivalently, you can use a dictionary comprehension (generally considered more readable, faster with huge data)
    # return {l[0]: l[1:].astype(numpy.float) for l in xyz_file}


def main():
    # parse the arguments
    parser = argparse.ArgumentParser(
        description="This script analyzes a user given xyz file and outputs the length of the bonds.")
    parser.add_argument("xyz_file", help="The filepath for the xyz file to analyze.")
    args = parser.parse_args()

    symbol_positions = open_xyz(args.xyz_file)

    # loop through the combinations of the keys
    for sym1, sym2 in itertools.combinations(symbol_positions, 2):
        bond_length_12 = calculate_distance(symbol_positions[sym1], symbol_positions[sym2])
        # PEP 8 test for trueness, not specific value True
        if bond_check(bond_length_12):
            print(F'{sym1} to {sym2} : {bond_length_12:.3f}')


if __name__ == '__main__':
    main()
