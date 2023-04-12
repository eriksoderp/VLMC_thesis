
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
import itertools
from Bio.SeqUtils import GC #gc_fraction

def dvstar_build(genome_path: Path, out_path: Path, threshold: float, min_count: int, max_depth: int):
    opath = ""
    out_path = out_path / Path(str(out_path) + f"_{threshold}_{min_count}_{max_depth}")
    out_path.mkdir(exist_ok=True)
    
    #desc = str(threshold) + '_' + str(min_count) + '_' + str(max_depth)
    for genome in os.listdir(genome_path):#tqdm(os.listdir(genome_path), desc=desc): #tqdm for development - useless on Bayes , #
        treename = get_bintree_name(genome, threshold, min_count, max_depth)
        opath = out_path / treename
        
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

def distances_par(out_path, tree1, tree2):
    args = (
        "./build/dvstar",
        "--mode",
        "dissimilarity",
        "--in-path",
        out_path / tree1,
        "--to-path",
        out_path / tree2
    )
    proc = subprocess.Popen(args, stdout=subprocess.PIPE)
    return (tree1, tree2, float(proc.stdout.readlines()[-1]))

def compare_trees(genome_path, out_path, threshold, min_count, max_depth):
    out_path = out_path / Path(str(out_path) + f"_{threshold}_{min_count}_{max_depth}")
    trees = os.listdir(out_path)
    #distances = []
    sizes = []
    i = 0

    distance_args = [(out_path,) + (tree1, tree2) for tree1, tree2 in list(itertools.combinations(trees, 2))]
    pool_obj3 = multiprocessing.Pool(32)
    distances = pool_obj3.starmap(distances_par, distance_args)
    pool_obj3.close()
    #for tree1, tree2 in list(itertools.combinations(trees, 2)):
    #    args = (
    #        "./build/dvstar",
    #        "--mode",
    #        "dissimilarity",
    #        "--in-path",
    #        out_path / tree1,
    #        "--to-path",
    #        out_path / tree2
    #    )
    #    proc = subprocess.Popen(args, stdout=subprocess.PIPE)
    #    distances.append((tree1, tree2, float(proc.stdout.readlines()[-1])))

    for tree in trees:
        args = (
            "./build/dvstar",
            "--mode",
            "size",
            "--in-path",
            out_path / tree
        )
        proc = subprocess.Popen(args, stdout=subprocess.PIPE)
        sizes.append((tree, int(proc.stdout.readlines()[1].decode("utf-8").split()[3][:-1])))

    columns = ['Tree1', 'Tree2', 'VLMC dist', 'threshold', 'min_count', 'max_depth']
    df = pd.DataFrame(columns=list(columns))

    dists = [(t1, t2, d) + (threshold, min_count, max_depth) for (t1, t2, d) in distances]

    for d in dists:
        df.loc[len(df)] = d

    df['Tree1 VLMC size'] = None
    df['Tree2 VLMC size'] = None
    for t, s in sizes:
        df.loc[df['Tree1'] == t, 'Tree1 VLMC size'] = s
        df.loc[df['Tree2'] == t, 'Tree2 VLMC size'] = s

    return df

def build(argv):
    genome_path = Path(argv[0])
    out_path = Path(argv[1])
    number_of_cores = int(argv[2])
    out_path.mkdir(exist_ok=True)

    combinations = [(genome_path, out_path,) + (threshold, min_count, max_depth) for threshold in (0.5, 3.9075) for min_count in (25,) for max_depth in (9, 12)]

    #pool_obj1 = multiprocessing.Pool(number_of_cores)
    #gcs = pool_obj1.starmap(dvstar_build, combinations)
    #pool_obj1.close()
    
    #pool_obj = multiprocessing.Pool(4)
    #dfs = pool_obj.starmap(compare_trees, combinations)
    #pool_obj.close()

    dfs = []
    for g, o, t, min_count, max_depth in combinations:
        dfs.append(compare_trees(g, o, t, min_count, max_depth))

    df = pd.concat(dfs, ignore_index=True)

    gcs = []
    for genome in os.listdir(genome_path):
        with open(genome_path / Path(genome), 'r') as f:
           new_gc = GC(f.read())
           gcs.append((genome[:-4], new_gc))

        f.close()

    df['Tree1 GC ratio'] = 0.0
    df['Tree2 GC ratio'] = 0.0
    for t, gc in gcs:
        df.loc[df['Tree1'].str.contains(t), 'Tree1 GC ratio'] = gc
        df.loc[df['Tree2'].str.contains(t), 'Tree2 GC ratio'] = gc

    for t, _ in gcs:
        df.loc[df['Tree1'].str.contains(t), 'Tree1'] = t
        df.loc[df['Tree2'].str.contains(t), 'Tree2'] = t

    df['Tree1'] = df['Tree1'].str.replace('_', ' ')
    df['Tree2'] = df['Tree2'].str.replace('_', ' ')

    df.to_csv('eukaryote_distances_25.csv')
    return

if __name__ == "__main__":
    build(sys.argv[1:])
