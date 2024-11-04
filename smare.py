"""Calculate sARE / sMARE
Usage: smare.py <csv file> <column number 1 (ground truth)> <column number 2>

Expected format of input CSV file: 
Column 1: QID
Remaining columns: QPP score for a particular method (1 method / column)
"""
from dataclasses import dataclass
import math
import numpy as np
import pandas as pd
import sys

@dataclass
class Pair:
    qid: str
    ap: float
    qpp_score: float
    ranks: list[int] # rank[0] = rank_ap, rank[1] = rank_qpp

def compute_ranks(l, f, index):
    l.sort(key=f)
    j, i = 0, 1
    rank, count = 1, 1
    while i < len(l) :
        if f(l[i]) == f(l[j]) :
            rank += i+1
            count += 1
        else:
            for k in range(j, i):
                l[k].ranks[index] = rank/count
            j, rank, count = i, i+1, 1
        i += 1
    for k in range(j, i):
        l[k].ranks[index] = rank/count


def smare(l1, l2):
    assert(len(l2) == (l := len(l1)))
    if l == 0: return 0 # corner case

    paired_list = [ Pair(q[0], q[1], q[2], [0, 0]) for q in zip(l1.index, l1, l2) ]
    compute_ranks(paired_list, lambda x: x.ap, 0)
    compute_ranks(paired_list, lambda x: x.qpp_score, 1)

    paired_list.sort(key=lambda x: x.qid)
    #for q in paired_list:
    #    print(f"{q.qid} {q.ap:>.6f} {q.qpp_score:>.6f} {q.ranks[0]:>5.2f} {q.ranks[1]:>5.2f}")

    sare = 0
    i = 1
    for p in paired_list:
        p.rank_qpp = i
        i += 1
        sare += abs(p.ranks[0] - p.ranks[1]) / l
    #print(paired_list)
    return sare/l


if __name__ == '__main__':
    # if len(sys.argv) != 3 :
    #     sys.exit('Usage: %s <csv file> <column number 1 (ground truth)> <column number 2>' % sys.argv[0])

    df = pd.read_csv(sys.argv[1], header=3, index_col=0)
    if sys.argv[2] == "100":
        ap_actual = df['ap@100']
    elif sys.argv[2] == "1000":
        ap_actual = df['ap@1000']
    else:
        print("Retrieval depth should be either either 100 or 1000.")
        exit(1)
    #ap_actual = df['ap']
    #ap_actual = df['ap@100']
    qpp_approaches = df.columns[1:] # first column ≡ ap
    for method in qpp_approaches:
        print(f'{method} {smare(ap_actual, df[method]):>.4f}')
