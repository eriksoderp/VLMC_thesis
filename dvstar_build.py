
import subprocess
from subprocess import check_output
import sys
from pathlib import Path
import pandas as pd 
import os 
from enum import Enum
import numpy as np  
import multiprocessing
import tempfile
from tqdm import tqdm
import re

def dvstar_build(genome_path: Path, out_path: Path, threshold: float, min_count: int, max_depth: int):
    opath = ""
    out_path = out_path / Path(str(out_path) + f"_{threshold}_{min_count}_{max_depth}")
    out_path.mkdir(exist_ok=True)

    desc = str(threshold) + '_' + str(min_count) + '_' + str(max_depth)
    for genome in tqdm(os.listdir(genome_path), desc=desc): #tqdm for development - useless on Bayes ,os.listdir(genome_path): #
        if "_1." in str(genome):
            opath = out_path / Path("original")
        else:
            opath = out_path / get_bintree_name(genome, threshold, min_count, max_depth)
        
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
            opath,
            "--in-or-out-of-core",
            "external"
        )
        subprocess.run(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)  


def get_bintree_name(genome_path: str, threshold: float, min_count: int, max_depth: int):
    return os.path.splitext(genome_path)[0] + f"_{threshold}_{min_count}_{max_depth}.bintree"

def compare_trees(genome_path, out_path, threshold, min_count, max_depth):
    out_path = out_path / Path(str(out_path) + f"_{threshold}_{min_count}_{max_depth}")
    trees = os.listdir(out_path)
    distances = []
    sizes = []
    trees.sort(key=lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])

    original = trees[0]
    print(trees)

    for i, tree in enumerate(trees[1:]):
        args = (
            "./build/dvstar",
            "--mode",
            "dissimilarity",
            "--in-path",
            out_path / original,
            "--to-path",
            out_path / tree
        )
        proc = subprocess.Popen(args, stdout=subprocess.PIPE)
        distances.append((i+1, float(proc.stdout.readlines()[-1])))

    for tree in trees:
        args = (
            "./build/dvstar",
            "--mode",
            "size",
            "--in-path",
            out_path / tree
        )
        proc = subprocess.Popen(args, stdout=subprocess.PIPE)
        sizes.append(int(proc.stdout.readlines()[1].decode("utf-8").split()[3][:-1]))

    original_size = sizes[0]
    return [(i, d, original_size, s) + (threshold, min_count, max_depth) for ((i, d), s) in zip(distances, sizes[1:])]

def build(argv):
    genome_path = Path(argv[0])
    out_path = Path(argv[1])
    number_of_cores = int(argv[2])
    out_path.mkdir(exist_ok=True)

    combinations = [(genome_path, out_path,) + (threshold, min_count, max_depth) for threshold in (0, 0.5, 1.2, 3.9075) for min_count in (2, 10, 25, 100) for max_depth in (9, 12)]

    pool_obj1 = multiprocessing.Pool(number_of_cores)
    pool_obj1.starmap(dvstar_build, combinations)
    pool_obj1.close()
    
    pool_obj = multiprocessing.Pool(number_of_cores)
    ans = pool_obj.starmap(compare_trees, combinations)
    pool_obj.close()

    columns = ['Sequence', 'VLMC dist', 'Original VLMC size', 'Mod VLMC size', 'threshold', 'min_count', 'max_depth']
    df = pd.DataFrame(columns=list(columns))
    for dists in ans:
        for sample in dists:
            df.loc[len(df)] = sample
    
    df_evo = pd.read_csv('modified_sequences/distances_wuhantest.csv')

    df = pd.merge(df, df_evo, on=['Sequence'])
    df.to_csv('vlmc_distances.csv')
    return

if __name__ == "__main__":
    build(sys.argv[1:])
