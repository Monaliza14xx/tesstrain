#!/usr/bin/env python3

# shuffle.py - shuffle lines in pseudo random order
#
# Usage:
#       shuffle.py [SEED [FILE]]
#
# Sort and shuffle the lines read from stdin in pseudo random order
# and write them to stdout.
#
# If FILE is given, then apply to that in-place (instead of stdin and stdout).
#
# The optional SEED argument is used as a seed for the random generator.
# A shuffled list can be reproduced by using the same seed again.

import random
import sys

# If at least one argument was given, the first argument is used as the seed.
if len(sys.argv) > 1:
    random.seed(sys.argv[1])

if len(sys.argv) > 2:
    input_file = sys.argv[2]
    # Read, sort and shuffle
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    # First sort the input lines (directory entries may come in undefined order).
    lines.sort()
    
    # Then shuffle the lines.
    random.shuffle(lines)
    
    # Write back to the same file
    with open(input_file, 'w') as f:
        f.writelines(lines)
else:
    # Read lines from standard input.
    lines = sys.stdin.readlines()
    
    # First sort the input lines (directory entries may come in undefined order).
    lines.sort()
    
    # Then shuffle the lines.
    random.shuffle(lines)
    
    # Write the shuffled lines to standard output.
    sys.stdout.writelines(lines)
