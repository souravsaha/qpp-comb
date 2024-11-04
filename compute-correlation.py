"""Calculate correlation based measures \kendall.. 
Usage: compute_correlation.py <csv file> <column number 1 (ground truth)> <column number 2>

Expected format of input CSV file: 
Column 1: QID
Remaining columns: QPP score for a particular method (1 method / column)
"""
from dataclasses import dataclass
import math
import numpy as np
import pandas as pd
import sys

if __name__ == '__main__':
    if len(sys.argv) != 3 :
        sys.exit('Usage: %s <csv file> <column number 1 (ground truth)> <column number 2 (retrieval depth)>' % sys.argv[0])

    df = pd.read_csv(sys.argv[1], header=3, index_col=0)
    print(df)

    if sys.argv[2] == "100":
        ap_actual = df['ap@100']
    elif sys.argv[2] == "1000":
        ap_actual = df['ap@1000']
    else:
        print("Retrieval depth should be 100 or 1000.")
        exit(1)

    # if sys.argv[2] == "trec678rb":
    #     ap_actual = df['ap']
    # elif sys.argv[2] == "trec678":
    #     ap_actual = df['ap@1000']
    # else :
    #     ap_actual = df['ap@100']
    
    qpp_approaches = df.columns[1:] # first column ≡ ap
    for method in qpp_approaches:
        print(f'Kendall {method} {df[method].corr(ap_actual, method = "kendall"):>.4f}')
        print(f'Pearson {method} {df[method].corr(ap_actual, method = "pearson"):>.4f}')
        #print(f'Kendall {method} {df[method].corr(ap_actual, method = "kendall"):>.4f}')
        #print(f'Spearman {method} {df[method].corr(ap_actual, method = "spearman"):>.4f}')


