#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pathlib
import sys


def split_file(input_file, ratio):
    """
    Splits a text file into list.train and  list.eval with lines ratio.
    Memory-optimized to handle large files efficiently.
    """
    if not isinstance(input_file, pathlib.Path):
        input_file = pathlib.Path(input_file)
    if not input_file.exists():
        print(f"'{input_file}' not exists!")
        return False
    
    # First pass: count lines to determine split point
    with open(input_file, 'r') as f:
        line_count = sum(1 for _ in f)
    
    split_point = int(ratio * line_count)
    output_dir = input_file.resolve().parent
    train_list = pathlib.Path(output_dir, 'list.train')
    eval_list = pathlib.Path(output_dir, 'list.eval')

    # Second pass: write lines to appropriate files
    with open(input_file, 'r') as fin, \
         open(train_list, 'w', newline='\n') as f1, \
         open(eval_list, 'w', newline='\n') as f2:
        for idx, line in enumerate(fin):
            line = line.rstrip('\n')
            if idx < split_point:
                f1.write(line)
                if idx < line_count - 1:  # Add newline for all but the last line of the entire file
                    f1.write('\n')
            else:
                f2.write(line)
                if idx < line_count - 1:  # Add newline for all but the last line of the entire file
                    f2.write('\n')
    return True


ratio = 0.95
input_file = None
if len(sys.argv) > 1:
    input_file = sys.argv[1]
if len(sys.argv) > 2:
    ratio = float(sys.argv[2])

split_file(input_file, ratio)
