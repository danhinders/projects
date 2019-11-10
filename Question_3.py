# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 10:04:14 2019

@author: danhi
"""

#Question3.py
#program to produce a plot of the funtion f(x) = e−x/p sin(πx) over the interval [0, 10],
# Date: 11/08/2019
# Author: Daniel Hinders

import matplotlib.pyplot as plt
import numpy as np
from numpy import exp, sin
from math import pi


def main():
# set a figure and title
    fig = plt.figure()
    fig.suptitle("Function f(x) = exp(-x/p)sinx(pi x), p = 1,2,5", fontsize = 12)

    # set x and y labels
    plt.xlabel("x")
    plt.ylabel("f(x)")
    
    # create array x
    x = np.arange(0., 10., 0.1)
    
    print(x)
    #parameter
    parameter = [1, 2, 5]
    
    # plot lines

    for p in parameter:
        # create label
        lab = "p="+ str(p)
        
        plt.plot((exp(-x/p)*sin(pi*x)), label = lab)
        
    # Place a legend to the right of the plot
    plt.legend(bbox_to_anchor=(0.675, 1), loc=2, borderaxespad=0.)
main()
