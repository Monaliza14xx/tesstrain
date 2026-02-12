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
    # Read from file
    with open(sys.argv[2], 'r') as fd0:
        lines = fd0.readlines()
else:
    # Read from stdin
    lines = sys.stdin.readlines()

# First sort the input lines (directory entries may come in undefined order).
lines.sort()

# Then shuffle the lines.
random.shuffle(lines)

if len(sys.argv) > 2:
    # Write to file
    with open(sys.argv[2], 'w') as fd1:
        fd1.writelines(lines)
else:
    # Write to stdout
    sys.stdout.writelines(lines)
