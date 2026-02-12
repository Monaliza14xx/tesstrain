#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pathlib
import sys


def split_file(input_file, ratio):
    """
    Splits a text file into list.train and  list.eval with lines ratio.
    """
    if not isinstance(input_file, pathlib.Path):
        input_file = pathlib.Path(input_file)
    if not input_file.exists():
        print(f"Error: '{input_file}' does not exist!")
        return False
    
    try:
        lines = input_file.read_text().splitlines()
    except (IOError, OSError) as e:
        print(f"Error reading file '{input_file}': {e}")
        return False

    if not lines:
        print(f"Warning: '{input_file}' is empty!")
        return False

    split_point = int(ratio * len(lines))
    output_dir = input_file.resolve().parent
    train_list = pathlib.Path(output_dir, 'list.train')
    eval_list = pathlib.Path(output_dir, 'list.eval')

    try:
        with open(train_list, 'w', newline='\n') as f1, open(
            eval_list, 'w', newline='\n'
        ) as f2:
            f1.write('\n'.join(lines[:split_point]))
            f2.write('\n'.join(lines[split_point:]))
    except (IOError, OSError) as e:
        print(f"Error writing output files: {e}")
        return False
    
    return True


ratio = 0.95
input_file = None
if len(sys.argv) > 1:
    input_file = sys.argv[1]
if len(sys.argv) > 2:
    ratio = float(sys.argv[2])

split_file(input_file, ratio)
