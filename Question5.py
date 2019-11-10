# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 16:37:35 2019

@author: danhi
"""

#Question5.py
#program to 
# Date: 11/08/2019
# Author: Daniel Hinders

import matplotlib.pyplot as plt
import numpy as np
from numpy import exp, sin, cos
from math import pi
import matplotlib.patches as mpatches


def main():
# set a figure and title
    fig = plt.figure()
    fig.suptitle("Projectile trajectories", fontsize = 12)

    # set x and y labels
    plt.xlabel("x = V * t * cos(theta)")
    plt.ylabel("y = V * t * sin(theta) - (g*t**2)/2")
    
    th = [1, 2, 3, 4, 5, 6, 7]
    
    theta0 = (pi/2)
    g = 9.8
    v = 100
    

    t_end = 2*v*np.sin(theta0) / g
    
    
    t = np.arange(0., t_end, 0.1)


    i = 0
    for j in th:
        theta = (pi/j)
        
        x = v*t*np.cos(theta)
        y = v*t*np.sin(theta)-(g*t**2)/2
        
        i = i + 1
        plt.plot(x, y, label = str(i) + "pi/16")
        
        
    plt.legend(bbox_to_anchor=(0.705, 1), loc=2, borderaxespad=0.)

main()
