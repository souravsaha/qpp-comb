#!/usr/bin/env ipython
# coding=utf-8

import sys
from math import *

if __name__ == '__main__':
    if len(sys.argv) != 3 :
        sys.exit('Usage: %s <rho> <number of observations over which rho is calculated>' % sys.argv[0])
    r = float(sys.argv[1])
    n = int(sys.argv[2])
    z = 0.5 * log((1 + r)/(1-r))
    three_sigma = 1.96 / sqrt(n-3)
    z_min = z - three_sigma
    z_max = z + three_sigma
    r_min = tanh(z_min)
    r_max = tanh(z_max)
    print(f'{r_min:>.2f} {r_max:>.2f}')
