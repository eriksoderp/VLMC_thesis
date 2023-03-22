import subprocess
import sys
from pathlib import Path
import pandas as pd 
import os 
from enum import Enum
import numpy as np  

def dvstar_build(genome_path: Path, out_path: Path, threshold: float, min_count: int, max_depth: int):
    first = True
    opath = ""
    out_path = out_path / Path(str(out_path) + f"_{threshold}_{min_count}_{max_depth}")
    out_path.mkdir(exist_ok=True)
    for genome in os.listdir(genome_path):
        print(genome)
        if first:
            opath = out_path / get_bintree_name("original", threshold, min_count, max_depth)
            first = False
        else:
            opath = out_path / get_bintree_name(genome, threshold, min_count, max_depth)

        """args = (
            "./build/src/pst-batch-training",
            genome_path / genome,
            "--min-count",
            str(min_count),
            "--max-depth",
            str(max_depth),
            "--threshold",
            str(threshold),
            "-o",
            opath
        )"""
        args = (
            "./build/dvstar",
            "--mode",
            "build",
            "--threshold",
            str(threshold),
            "--min-count",
            str(min_count),
            "--max-depth",
            str(max_depth),
            "--fasta-path",
            genome_path / genome,
            "--out-path",
            opath
        )
        subprocess.run(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def get_bintree_name(genome_path: str, threshold: float, min_count: int, max_depth: int):
    return os.path.splitext(genome_path)[0] + f"_{threshold}_{min_count}_{max_depth}.bintree"

def build(argv):
    genome_path = Path(argv[0])
    out_path = Path(argv[1])
    out_path.mkdir(exist_ok=True)

    combinations = [(threshold, min_count, max_depth) for threshold in (0, 0.5, 1.2, 3.9075) for min_count in (2, 10, 100) for max_depth in (9, 12)]

    for (threshold, min_count, max_depth) in combinations:
        dvstar_build(genome_path, out_path, threshold, min_count, max_depth)

    return

if __name__ == "__main__":
    build(sys.argv[1:])
